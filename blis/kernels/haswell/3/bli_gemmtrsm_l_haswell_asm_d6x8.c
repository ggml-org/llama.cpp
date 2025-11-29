/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#define SGEMM_OUTPUT_GS_BETA_NZ \
	vextractf128(imm(1), ymm0, xmm2) \
	vmovss(xmm0, mem(rcx)) \
	vpermilps(imm(0x39), xmm0, xmm1) \
	vmovss(xmm1, mem(rcx, rsi, 1)) \
	vpermilps(imm(0x39), xmm1, xmm0) \
	vmovss(xmm0, mem(rcx, rsi, 2)) \
	vpermilps(imm(0x39), xmm0, xmm1) \
	vmovss(xmm1, mem(rcx, r13, 1)) \
	vmovss(xmm2, mem(rcx, rsi, 4)) \
	vpermilps(imm(0x39), xmm2, xmm1) \
	vmovss(xmm1, mem(rcx, r15, 1)) \
	vpermilps(imm(0x39), xmm1, xmm2) \
	vmovss(xmm2, mem(rcx, r13, 2)) \
	vpermilps(imm(0x39), xmm2, xmm1) \
	vmovss(xmm1, mem(rcx, r10, 1))


void bli_sgemmtrsm_l_haswell_asm_6x16
     (
             dim_t      m, \
             dim_t      n, \
             dim_t      k0, \
       const void*      alpha, \
       const void*      a10, \
       const void*      a11, \
       const void*      b01, \
             void*      b11, \
             void*      c11, inc_t rs_c0, inc_t cs_c0, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	float*   beta   = bli_sm1;

	GEMMTRSM_UKR_SETUP_CT_ANY( s, 6, 16, true );

	begin_asm()

	vzeroall() // zero all xmm/ymm registers.


	mov(var(a10), rax) // load address of a.
	mov(var(b01), rbx) // load address of b.

	add(imm(32*4), rbx)
	 // initialize loop by pre-loading
	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)

	mov(var(b11), rcx) // load address of b11
	mov(imm(16), rdi) // set rs_b = PACKNR = 16
	lea(mem(, rdi, 4), rdi) // rs_b *= sizeof(float)

	 // NOTE: c11, rs_c, and cs_c aren't
	 // needed for a while, but we load
	 // them now to avoid stalling later.
	mov(var(c11), r8) // load address of c11
	mov(var(rs_c), r9) // load rs_c
	lea(mem(, r9 , 4), r9) // rs_c *= sizeof(float)
	mov(var(k_left)0, r10) // load cs_c
	lea(mem(, r10, 4), r10) // cs_c *= sizeof(float)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.SCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.SLOOPKITER) // MAIN LOOP


	 // iteration 0
	prefetch(0, mem(rax, 64*4))

	vbroadcastss(mem(rax, 0*4), ymm2)
	vbroadcastss(mem(rax, 1*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 2*4), ymm2)
	vbroadcastss(mem(rax, 3*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 4*4), ymm2)
	vbroadcastss(mem(rax, 5*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, -2*32), ymm0)
	vmovaps(mem(rbx, -1*32), ymm1)

	 // iteration 1
	vbroadcastss(mem(rax, 6*4), ymm2)
	vbroadcastss(mem(rax, 7*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 8*4), ymm2)
	vbroadcastss(mem(rax, 9*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 10*4), ymm2)
	vbroadcastss(mem(rax, 11*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, 0*32), ymm0)
	vmovaps(mem(rbx, 1*32), ymm1)

	 // iteration 2
	prefetch(0, mem(rax, 76*4))

	vbroadcastss(mem(rax, 12*4), ymm2)
	vbroadcastss(mem(rax, 13*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 14*4), ymm2)
	vbroadcastss(mem(rax, 15*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 16*4), ymm2)
	vbroadcastss(mem(rax, 17*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, 2*32), ymm0)
	vmovaps(mem(rbx, 3*32), ymm1)

	 // iteration 3
	vbroadcastss(mem(rax, 18*4), ymm2)
	vbroadcastss(mem(rax, 19*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 20*4), ymm2)
	vbroadcastss(mem(rax, 21*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 22*4), ymm2)
	vbroadcastss(mem(rax, 23*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(imm(4*6*4), rax) // a += 4*6  (unroll x mr)
	add(imm(4*16*4), rbx) // b += 4*16 (unroll x nr)

	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.SLOOPKITER) // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.SPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT) // EDGE LOOP

	prefetch(0, mem(rax, 64*4))

	vbroadcastss(mem(rax, 0*4), ymm2)
	vbroadcastss(mem(rax, 1*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 2*4), ymm2)
	vbroadcastss(mem(rax, 3*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 4*4), ymm2)
	vbroadcastss(mem(rax, 5*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(imm(1*6*4), rax) // a += 1*6  (unroll x mr)
	add(imm(1*16*4), rbx) // b += 1*16 (unroll x nr)

	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.SLOOPKLEFT) // iterate again if i != 0.



	label(.SPOSTACCUM)

	 // ymm4..ymm15 = -a10 * b01



	mov(var(alpha), rbx) // load address of alpha
	vbroadcastss(mem(rbx), ymm3) // load alpha and duplicate




	mov(imm(1), rsi) // load cs_b = 1
	lea(mem(, rsi, 4), rsi) // cs_b *= sizeof(float)

	lea(mem(rcx, rsi, 8), rdx) // load address of b11 + 8*cs_b

	mov(rcx, r11) // save rcx = b11        for later
	mov(rdx, r14) // save rdx = b11+8*cs_b for later


	 // b11 := alpha * b11 - a10 * b01
	vfmsub231ps(mem(rcx), ymm3, ymm4)
	add(rdi, rcx)
	vfmsub231ps(mem(rdx), ymm3, ymm5)
	add(rdi, rdx)

	vfmsub231ps(mem(rcx), ymm3, ymm6)
	add(rdi, rcx)
	vfmsub231ps(mem(rdx), ymm3, ymm7)
	add(rdi, rdx)

	vfmsub231ps(mem(rcx), ymm3, ymm8)
	add(rdi, rcx)
	vfmsub231ps(mem(rdx), ymm3, ymm9)
	add(rdi, rdx)

	vfmsub231ps(mem(rcx), ymm3, ymm10)
	add(rdi, rcx)
	vfmsub231ps(mem(rdx), ymm3, ymm11)
	add(rdi, rdx)

	vfmsub231ps(mem(rcx), ymm3, ymm12)
	add(rdi, rcx)
	vfmsub231ps(mem(rdx), ymm3, ymm13)
	add(rdi, rdx)

	vfmsub231ps(mem(rcx), ymm3, ymm14)
	//add(rdi, rcx)
	vfmsub231ps(mem(rdx), ymm3, ymm15)
	//add(rdi, rdx)



	 // prefetch c11

#if 0
	mov(r8, rcx) // load address of c11 from r8
	 // Note: r9 = rs_c * sizeof(float)

	lea(mem(r9 , r9 , 2), r13) // r13 = 3*rs_c;
	lea(mem(rcx, r13, 1), rdx) // rdx = c11 + 3*rs_c;

	prefetch(0, mem(rcx, 0*8)) // prefetch c11 + 0*rs_c
	prefetch(0, mem(rcx, r9, 1, 0*8)) // prefetch c11 + 1*rs_c
	prefetch(0, mem(rcx, r9 , 2, 0*8)) // prefetch c11 + 2*rs_c
	prefetch(0, mem(rdx, 0*8)) // prefetch c11 + 3*rs_c
	prefetch(0, mem(rdx, r9, 1, 0*8)) // prefetch c11 + 4*rs_c
	prefetch(0, mem(rdx, r9 , 2, 0*8)) // prefetch c11 + 5*rs_c
#endif




	 // trsm computation begins here

	 // Note: contents of b11 are stored as
	 // ymm4  ymm5  = ( beta00..07 ) ( beta08..0F )
	 // ymm6  ymm7  = ( beta10..17 ) ( beta18..1F )
	 // ymm8  ymm9  = ( beta20..27 ) ( beta28..2F )
	 // ymm10 ymm11 = ( beta30..37 ) ( beta38..3F )
	 // ymm12 ymm13 = ( beta40..47 ) ( beta48..4F )
	 // ymm14 ymm15 = ( beta50..57 ) ( beta58..5F )


	mov(var(a11), rax) // load address of a11

	mov(r11, rcx) // recall address of b11
	mov(r14, rdx) // recall address of b11+8*cs_b
	 // Note: rdi = rs_b

	 // iteration 0 -------------

	vbroadcastss(mem(0+0*6)*4(rax), ymm0) // ymm0 = (1/alpha00)

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulps(ymm0, ymm4, ymm4) // ymm4 *= (1/alpha00)
	vmulps(ymm0, ymm5, ymm5) // ymm5 *= (1/alpha00)
#else
	vdivps(ymm0, ymm4, ymm4) // ymm4 /= alpha00
	vdivps(ymm0, ymm5, ymm5) // ymm5 /= alpha00
#endif

	vmovups(ymm4, mem(rcx)) // store ( beta00..beta07 ) = ymm4
	vmovups(ymm5, mem(rdx)) // store ( beta08..beta0F ) = ymm5
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 1 -------------

	vbroadcastss(mem(1+0*6)*4(rax), ymm0) // ymm0 = alpha10
	vbroadcastss(mem(1+1*6)*4(rax), ymm1) // ymm1 = (1/alpha11)

	vmulps(ymm0, ymm4, ymm2) // ymm2 = alpha10 * ymm4
	vmulps(ymm0, ymm5, ymm3) // ymm3 = alpha10 * ymm5

	vsubps(ymm2, ymm6, ymm6) // ymm6 -= ymm2
	vsubps(ymm3, ymm7, ymm7) // ymm7 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulps(ymm1, ymm6, ymm6) // ymm6 *= (1/alpha11)
	vmulps(ymm1, ymm7, ymm7) // ymm7 *= (1/alpha11)
#else
	vdivps(ymm1, ymm6, ymm6) // ymm6 /= alpha11
	vdivps(ymm1, ymm7, ymm7) // ymm7 /= alpha11
#endif

	vmovups(ymm6, mem(rcx)) // store ( beta10..beta17 ) = ymm6
	vmovups(ymm7, mem(rdx)) // store ( beta18..beta1F ) = ymm7
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 2 -------------

	vbroadcastss(mem(2+0*6)*4(rax), ymm0) // ymm0 = alpha20
	vbroadcastss(mem(2+1*6)*4(rax), ymm1) // ymm1 = alpha21

	vmulps(ymm0, ymm4, ymm2) // ymm2 = alpha20 * ymm4
	vmulps(ymm0, ymm5, ymm3) // ymm3 = alpha20 * ymm5

	vbroadcastss(mem(2+2*6)*4(rax), ymm0) // ymm0 = (1/alpha22)

	vfmadd231ps(ymm1, ymm6, ymm2) // ymm2 += alpha21 * ymm6
	vfmadd231ps(ymm1, ymm7, ymm3) // ymm3 += alpha21 * ymm7

	vsubps(ymm2, ymm8, ymm8) // ymm8 -= ymm2
	vsubps(ymm3, ymm9, ymm9) // ymm9 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulps(ymm0, ymm8, ymm8) // ymm8 *= (1/alpha22)
	vmulps(ymm0, ymm9, ymm9) // ymm9 *= (1/alpha22)
#else
	vdivps(ymm0, ymm8, ymm8) // ymm8 /= alpha22
	vdivps(ymm0, ymm9, ymm9) // ymm9 /= alpha22
#endif

	vmovups(ymm8, mem(rcx)) // store ( beta20..beta27 ) = ymm8
	vmovups(ymm9, mem(rdx)) // store ( beta28..beta2F ) = ymm9
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 3 -------------

	vbroadcastss(mem(3+0*6)*4(rax), ymm0) // ymm0 = alpha30
	vbroadcastss(mem(3+1*6)*4(rax), ymm1) // ymm1 = alpha31

	vmulps(ymm0, ymm4, ymm2) // ymm2 = alpha30 * ymm4
	vmulps(ymm0, ymm5, ymm3) // ymm3 = alpha30 * ymm5

	vbroadcastss(mem(3+2*6)*4(rax), ymm0) // ymm0 = alpha32

	vfmadd231ps(ymm1, ymm6, ymm2) // ymm2 += alpha31 * ymm6
	vfmadd231ps(ymm1, ymm7, ymm3) // ymm3 += alpha31 * ymm7

	vbroadcastss(mem(3+3*6)*4(rax), ymm1) // ymm0 = (1/alpha33)

	vfmadd231ps(ymm0, ymm8, ymm2) // ymm2 += alpha32 * ymm8
	vfmadd231ps(ymm0, ymm9, ymm3) // ymm3 += alpha32 * ymm9

	vsubps(ymm2, ymm10, ymm10) // ymm10 -= ymm2
	vsubps(ymm3, ymm11, ymm11) // ymm11 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulps(ymm1, ymm10, ymm10) // ymm10 *= (1/alpha33)
	vmulps(ymm1, ymm11, ymm11) // ymm11 *= (1/alpha33)
#else
	vdivps(ymm1, ymm10, ymm10) // ymm10 /= alpha33
	vdivps(ymm1, ymm11, ymm11) // ymm11 /= alpha33
#endif

	vmovups(ymm10, mem(rcx)) // store ( beta30..beta37 ) = ymm10
	vmovups(ymm11, mem(rdx)) // store ( beta38..beta3F ) = ymm11
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 4 -------------

	vbroadcastss(mem(4+0*6)*4(rax), ymm0) // ymm0 = alpha40
	vbroadcastss(mem(4+1*6)*4(rax), ymm1) // ymm1 = alpha41

	vmulps(ymm0, ymm4, ymm2) // ymm2 = alpha40 * ymm4
	vmulps(ymm0, ymm5, ymm3) // ymm3 = alpha40 * ymm5

	vbroadcastss(mem(4+2*6)*4(rax), ymm0) // ymm0 = alpha42

	vfmadd231ps(ymm1, ymm6, ymm2) // ymm2 += alpha41 * ymm6
	vfmadd231ps(ymm1, ymm7, ymm3) // ymm3 += alpha41 * ymm7

	vbroadcastss(mem(4+3*6)*4(rax), ymm1) // ymm1 = alpha43

	vfmadd231ps(ymm0, ymm8, ymm2) // ymm2 += alpha42 * ymm8
	vfmadd231ps(ymm0, ymm9, ymm3) // ymm3 += alpha42 * ymm9

	vbroadcastss(mem(4+4*6)*4(rax), ymm0) // ymm0 = (1/alpha44)

	vfmadd231ps(ymm1, ymm10, ymm2) // ymm2 += alpha43 * ymm10
	vfmadd231ps(ymm1, ymm11, ymm3) // ymm3 += alpha43 * ymm11

	vsubps(ymm2, ymm12, ymm12) // ymm12 -= ymm2
	vsubps(ymm3, ymm13, ymm13) // ymm13 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulps(ymm0, ymm12, ymm12) // ymm12 *= (1/alpha44)
	vmulps(ymm0, ymm13, ymm13) // ymm13 *= (1/alpha44)
#else
	vdivps(ymm0, ymm12, ymm12) // ymm12 /= alpha44
	vdivps(ymm0, ymm13, ymm13) // ymm13 /= alpha44
#endif

	vmovups(ymm12, mem(rcx)) // store ( beta40..beta47 ) = ymm12
	vmovups(ymm13, mem(rdx)) // store ( beta48..beta4F ) = ymm13
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 5 -------------

	vbroadcastss(mem(5+0*6)*4(rax), ymm0) // ymm0 = alpha50
	vbroadcastss(mem(5+1*6)*4(rax), ymm1) // ymm1 = alpha51

	vmulps(ymm0, ymm4, ymm2) // ymm2 = alpha50 * ymm4
	vmulps(ymm0, ymm5, ymm3) // ymm3 = alpha50 * ymm5

	vbroadcastss(mem(5+2*6)*4(rax), ymm0) // ymm0 = alpha52

	vfmadd231ps(ymm1, ymm6, ymm2) // ymm2 += alpha51 * ymm6
	vfmadd231ps(ymm1, ymm7, ymm3) // ymm3 += alpha51 * ymm7

	vbroadcastss(mem(5+3*6)*4(rax), ymm1) // ymm1 = alpha53

	vfmadd231ps(ymm0, ymm8, ymm2) // ymm2 += alpha52 * ymm8
	vfmadd231ps(ymm0, ymm9, ymm3) // ymm3 += alpha52 * ymm9

	vbroadcastss(mem(5+4*6)*4(rax), ymm0) // ymm0 = alpha54

	vfmadd231ps(ymm1, ymm10, ymm2) // ymm2 += alpha53 * ymm10
	vfmadd231ps(ymm1, ymm11, ymm3) // ymm3 += alpha53 * ymm11

	vbroadcastss(mem(5+5*6)*4(rax), ymm1) // ymm1 = (1/alpha55)

	vfmadd231ps(ymm0, ymm12, ymm2) // ymm2 += alpha54 * ymm12
	vfmadd231ps(ymm0, ymm13, ymm3) // ymm3 += alpha54 * ymm13

	vsubps(ymm2, ymm14, ymm14) // ymm14 -= ymm2
	vsubps(ymm3, ymm15, ymm15) // ymm15 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulps(ymm1, ymm14, ymm14) // ymm14 *= (1/alpha55)
	vmulps(ymm1, ymm15, ymm15) // ymm15 *= (1/alpha55)
#else
	vdivps(ymm1, ymm14, ymm14) // ymm14 /= alpha55
	vdivps(ymm1, ymm15, ymm15) // ymm15 /= alpha55
#endif

	vmovups(ymm14, mem(rcx)) // store ( beta50..beta57 ) = ymm14
	vmovups(ymm15, mem(rdx)) // store ( beta58..beta5F ) = ymm15
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b





	mov(r8, rcx) // load address of c11 from r8
	mov(r9, rdi) // load rs_c (in bytes) from r9
	mov(r10, rsi) // load cs_c (in bytes) from r10

	lea(mem(rcx, rsi, 8), rdx) // load address of c11 + 8*cs_c;
	lea(mem(rcx, rdi, 4), r14) // load address of c11 + 4*rs_c;

	 // These are used in the macros below.
	lea(mem(rsi, rsi, 2), r13) // r13 = 3*cs_c;
	lea(mem(rsi, rsi, 4), r15) // r15 = 5*cs_c;
	lea(mem(r13, rsi, 4), r10) // r10 = 7*cs_c;



	cmp(imm(4), rsi) // set ZF if (4*cs_c) == 4.
	jz(.SROWSTORED) // jump to row storage case



	cmp(imm(4), rdi) // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED) // jump to column storage case



	 // if neither row- or column-
	 // stored, use general case.
	label(.SGENSTORED)


	vmovaps(ymm4, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm6, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm8, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm10, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm12, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm14, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ


	mov(rdx, rcx) // rcx = c11 + 8*cs_c


	vmovaps(ymm5, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm7, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm9, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm11, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm13, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovaps(ymm15, ymm0)
	SGEMM_OUTPUT_GS_BETA_NZ



	jmp(.SDONE)



	label(.SROWSTORED)


	vmovups(ymm4, mem(rcx))
	add(rdi, rcx)
	vmovups(ymm5, mem(rdx))
	add(rdi, rdx)

	vmovups(ymm6, mem(rcx))
	add(rdi, rcx)
	vmovups(ymm7, mem(rdx))
	add(rdi, rdx)

	vmovups(ymm8, mem(rcx))
	add(rdi, rcx)
	vmovups(ymm9, mem(rdx))
	add(rdi, rdx)

	vmovups(ymm10, mem(rcx))
	add(rdi, rcx)
	vmovups(ymm11, mem(rdx))
	add(rdi, rdx)

	vmovups(ymm12, mem(rcx))
	add(rdi, rcx)
	vmovups(ymm13, mem(rdx))
	add(rdi, rdx)

	vmovups(ymm14, mem(rcx))
	//add(rdi, rcx)
	vmovups(ymm15, mem(rdx))
	//add(rdi, rdx)


	jmp(.SDONE)



	label(.SCOLSTORED)


	vunpcklps(ymm6, ymm4, ymm0)
	vunpcklps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
	vmovups(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
	vmovups(xmm3, mem(rcx, r15, 1)) // store ( gamma05..gamma35 )


	vunpckhps(ymm6, ymm4, ymm0)
	vunpckhps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovups(xmm0, mem(rcx, rsi, 2)) // store ( gamma02..gamma32 )
	vmovups(xmm1, mem(rcx, r13, 1)) // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, r13, 2)) // store ( gamma06..gamma36 )
	vmovups(xmm3, mem(rcx, r10, 1)) // store ( gamma07..gamma37 )

	lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c

	vunpcklps(ymm14, ymm12, ymm0)
	vunpckhps(ymm14, ymm12, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovlpd(xmm0, mem(r14)) // store ( gamma40..gamma50 )
	vmovhpd(xmm0, mem(r14, rsi, 1)) // store ( gamma41..gamma51 )
	vmovlpd(xmm1, mem(r14, rsi, 2)) // store ( gamma42..gamma52 )
	vmovhpd(xmm1, mem(r14, r13, 1)) // store ( gamma43..gamma53 )
	vmovlpd(xmm2, mem(r14, rsi, 4)) // store ( gamma44..gamma54 )
	vmovhpd(xmm2, mem(r14, r15, 1)) // store ( gamma45..gamma55 )
	vmovlpd(xmm3, mem(r14, r13, 2)) // store ( gamma46..gamma56 )
	vmovhpd(xmm3, mem(r14, r10, 1)) // store ( gamma47..gamma57 )

	lea(mem(r14, rsi, 8), r14) // r14 += 8*cs_c


	vunpcklps(ymm7, ymm5, ymm0)
	vunpcklps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovups(xmm0, mem(rcx)) // store ( gamma08..gamma38 )
	vmovups(xmm1, mem(rcx, rsi, 1)) // store ( gamma09..gamma39 )
	vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma0C..gamma3C )
	vmovups(xmm3, mem(rcx, r15, 1)) // store ( gamma0D..gamma3D )

	vunpckhps(ymm7, ymm5, ymm0)
	vunpckhps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovups(xmm0, mem(rcx, rsi, 2)) // store ( gamma0A..gamma3A )
	vmovups(xmm1, mem(rcx, r13, 1)) // store ( gamma0B..gamma3B )
	vmovups(xmm2, mem(rcx, r13, 2)) // store ( gamma0E..gamma3E )
	vmovups(xmm3, mem(rcx, r10, 1)) // store ( gamma0F..gamma3F )

	//lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c

	vunpcklps(ymm15, ymm13, ymm0)
	vunpckhps(ymm15, ymm13, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovlpd(xmm0, mem(r14)) // store ( gamma48..gamma58 )
	vmovhpd(xmm0, mem(r14, rsi, 1)) // store ( gamma49..gamma59 )
	vmovlpd(xmm1, mem(r14, rsi, 2)) // store ( gamma4A..gamma5A )
	vmovhpd(xmm1, mem(r14, r13, 1)) // store ( gamma4B..gamma5B )
	vmovlpd(xmm2, mem(r14, rsi, 4)) // store ( gamma4C..gamma5C )
	vmovhpd(xmm2, mem(r14, r15, 1)) // store ( gamma4D..gamma5D )
	vmovlpd(xmm3, mem(r14, r13, 2)) // store ( gamma4E..gamma5E )
	vmovhpd(xmm3, mem(r14, r10, 1)) // store ( gamma4F..gamma5F )

	//lea(mem(r14, rsi, 8), r14) // r14 += 8*cs_c




	label(.SDONE)

	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	  [k_iter] "m" (k_iter), // 0
	  [k_left] "m" (k_left), // 1
	  [a10]    "m" (a10),    // 2
	  [b01]    "m" (b01),    // 3
	  [beta]   "m" (beta),   // 4
	  [alpha]  "m" (alpha),  // 5
	  [a11]    "m" (a11),    // 6
	  [b11]    "m" (b11),    // 7
	  [c11]    "m" (c11),    // 8
	  [rs_c]   "m" (rs_c),   // 9
	  [cs_c]   "m" (cs_c)    // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMMTRSM_UKR_FLUSH_CT( s );
}



#define DGEMM_OUTPUT_GS_BETA_NZ \
	vextractf128(imm(1), ymm0, xmm1) \
	vmovlpd(xmm0, mem(rcx)) \
	vmovhpd(xmm0, mem(rcx, rsi, 1)) \
	vmovlpd(xmm1, mem(rcx, rsi, 2)) \
	vmovhpd(xmm1, mem(rcx, r13, 1)) /*\
	vextractf128(imm(1), ymm2, xmm1) \
	vmovlpd(xmm2, mem(rcx, rsi, 4)) \
	vmovhpd(xmm2, mem(rcx, r15, 1)) \
	vmovlpd(xmm1, mem(rcx, r13, 2)) \
	vmovhpd(xmm1, mem(rcx, r10, 1))*/

void bli_dgemmtrsm_l_haswell_asm_6x8
     (
             dim_t      m, \
             dim_t      n, \
             dim_t      k0, \
       const void*      alpha, \
       const void*      a10, \
       const void*      a11, \
       const void*      b01, \
             void*      b11, \
             void*      c11, inc_t rs_c0, inc_t cs_c0, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	double*  beta   = bli_dm1;

	GEMMTRSM_UKR_SETUP_CT_ANY( d, 6, 8, true );

	begin_asm()

	vzeroall() // zero all xmm/ymm registers.


	mov(var(a10), rax) // load address of a.
	mov(var(b01), rbx) // load address of b.

	add(imm(32*4), rbx)
	 // initialize loop by pre-loading
	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)

	mov(var(b11), rcx) // load address of b11
	mov(imm(8), rdi) // set rs_b = PACKNR = 8
	lea(mem(, rdi, 8), rdi) // rs_b *= sizeof(double)

	 // NOTE: c11, rs_c, and cs_c aren't
	 // needed for a while, but we load
	 // them now to avoid stalling later.
	mov(var(c11), r8) // load address of c11
	mov(var(rs_c), r9) // load rs_c
	lea(mem(, r9 , 8), r9) // rs_c *= sizeof(double)
	mov(var(k_left)0, r10) // load cs_c
	lea(mem(, r10, 8), r10) // cs_c *= sizeof(double)



	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.DCONSIDKLEFT) // if i == 0, jump to code that
	 // contains the k_left loop.


	label(.DLOOPKITER) // MAIN LOOP


	 // iteration 0
	prefetch(0, mem(rax, 64*8))

	vbroadcastsd(mem(rax, 0*8), ymm2)
	vbroadcastsd(mem(rax, 1*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 2*8), ymm2)
	vbroadcastsd(mem(rax, 3*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 4*8), ymm2)
	vbroadcastsd(mem(rax, 5*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, -2*32), ymm0)
	vmovapd(mem(rbx, -1*32), ymm1)

	 // iteration 1
	prefetch(0, mem(rax, 72*8))

	vbroadcastsd(mem(rax, 6*8), ymm2)
	vbroadcastsd(mem(rax, 7*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 8*8), ymm2)
	vbroadcastsd(mem(rax, 9*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 10*8), ymm2)
	vbroadcastsd(mem(rax, 11*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, 0*32), ymm0)
	vmovapd(mem(rbx, 1*32), ymm1)

	 // iteration 2
	prefetch(0, mem(rax, 80*8))

	vbroadcastsd(mem(rax, 12*8), ymm2)
	vbroadcastsd(mem(rax, 13*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 14*8), ymm2)
	vbroadcastsd(mem(rax, 15*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 16*8), ymm2)
	vbroadcastsd(mem(rax, 17*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, 2*32), ymm0)
	vmovapd(mem(rbx, 3*32), ymm1)

	 // iteration 3
	vbroadcastsd(mem(rax, 18*8), ymm2)
	vbroadcastsd(mem(rax, 19*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 20*8), ymm2)
	vbroadcastsd(mem(rax, 21*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 22*8), ymm2)
	vbroadcastsd(mem(rax, 23*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	add(imm(4*6*8), rax) // a += 4*6 (unroll x mr)
	add(imm(4*8*8), rbx) // b += 4*8 (unroll x nr)

	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.DLOOPKITER) // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.DPOSTACCUM) // if i == 0, we're done; jump to end.
	 // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT) // EDGE LOOP

	prefetch(0, mem(rax, 64*8))

	vbroadcastsd(mem(rax, 0*8), ymm2)
	vbroadcastsd(mem(rax, 1*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 2*8), ymm2)
	vbroadcastsd(mem(rax, 3*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 4*8), ymm2)
	vbroadcastsd(mem(rax, 5*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	add(imm(1*6*8), rax) // a += 1*6 (unroll x mr)
	add(imm(1*8*8), rbx) // b += 1*8 (unroll x nr)

	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.DLOOPKLEFT) // iterate again if i != 0.



	label(.DPOSTACCUM)

	 // ymm4..ymm15 = -a10 * b01




	mov(var(alpha), rbx) // load address of alpha
	vbroadcastsd(mem(rbx), ymm3) // load alpha and duplicate




	mov(imm(1), rsi) // set cs_b = 1
	lea(mem(, rsi, 8), rsi) // cs_b *= sizeof(double)

	lea(mem(rcx, rsi, 4), rdx) // load address of b11 + 4*cs_b

	mov(rcx, r11) // save rcx = b11        for later
	mov(rdx, r14) // save rdx = b11+4*cs_b for later


	 // b11 := alpha * b11 - a10 * b01
	vfmsub231pd(mem(rcx), ymm3, ymm4)
	add(rdi, rcx)
	vfmsub231pd(mem(rdx), ymm3, ymm5)
	add(rdi, rdx)

	vfmsub231pd(mem(rcx), ymm3, ymm6)
	add(rdi, rcx)
	vfmsub231pd(mem(rdx), ymm3, ymm7)
	add(rdi, rdx)

	vfmsub231pd(mem(rcx), ymm3, ymm8)
	add(rdi, rcx)
	vfmsub231pd(mem(rdx), ymm3, ymm9)
	add(rdi, rdx)

	vfmsub231pd(mem(rcx), ymm3, ymm10)
	add(rdi, rcx)
	vfmsub231pd(mem(rdx), ymm3, ymm11)
	add(rdi, rdx)

	vfmsub231pd(mem(rcx), ymm3, ymm12)
	add(rdi, rcx)
	vfmsub231pd(mem(rdx), ymm3, ymm13)
	add(rdi, rdx)

	vfmsub231pd(mem(rcx), ymm3, ymm14)
  //add(rdi, rcx)
	vfmsub231pd(mem(rdx), ymm3, ymm15)
  //add(rdi, rdx)



	 // prefetch c11

#if 0
	mov(r8, rcx) // load address of c11 from r8
	 // Note: r9 = rs_c * sizeof(double)

	lea(mem(r9 , r9 , 2), r13) // r13 = 3*rs_c;
	lea(mem(rcx, r13, 1), rdx) // rdx = c11 + 3*rs_c;

	prefetch(0, mem(rcx, 7*8)) // prefetch c11 + 0*rs_c
	prefetch(0, mem(rcx, r9, 1, 7*8)) // prefetch c11 + 1*rs_c
	prefetch(0, mem(rcx, r9 , 2, 7*8)) // prefetch c11 + 2*rs_c
	prefetch(0, mem(rdx, 7*8)) // prefetch c11 + 3*rs_c
	prefetch(0, mem(rdx, r9, 1, 7*8)) // prefetch c11 + 4*rs_c
	prefetch(0, mem(rdx, r9 , 2, 7*8)) // prefetch c11 + 5*rs_c
#endif




	 // trsm computation begins here

	 // Note: contents of b11 are stored as
	 // ymm4  ymm5  = ( beta00..03 ) ( beta04..07 )
	 // ymm6  ymm7  = ( beta10..13 ) ( beta14..17 )
	 // ymm8  ymm9  = ( beta20..23 ) ( beta24..27 )
	 // ymm10 ymm11 = ( beta30..33 ) ( beta34..37 )
	 // ymm12 ymm13 = ( beta40..43 ) ( beta44..47 )
	 // ymm14 ymm15 = ( beta50..53 ) ( beta54..57 )


	mov(var(a11), rax) // load address of a11

	mov(r11, rcx) // recall address of b11
	mov(r14, rdx) // recall address of b11+4*cs_b
	 // Note: rdi = rs_b

	 // iteration 0 -------------

	vbroadcastsd(mem(0+0*6)*8(rax), ymm0) // ymm0 = (1/alpha00)

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulpd(ymm0, ymm4, ymm4) // ymm4 *= (1/alpha00)
	vmulpd(ymm0, ymm5, ymm5) // ymm5 *= (1/alpha00)
#else
	vdivpd(ymm0, ymm4, ymm4) // ymm4 /= alpha00
	vdivpd(ymm0, ymm5, ymm5) // ymm5 /= alpha00
#endif

	vmovupd(ymm4, mem(rcx)) // store ( beta00..beta03 ) = ymm4
	vmovupd(ymm5, mem(rdx)) // store ( beta04..beta07 ) = ymm5
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 1 -------------

	vbroadcastsd(mem(1+0*6)*8(rax), ymm0) // ymm0 = alpha10
	vbroadcastsd(mem(1+1*6)*8(rax), ymm1) // ymm1 = (1/alpha11)

	vmulpd(ymm0, ymm4, ymm2) // ymm2 = alpha10 * ymm4
	vmulpd(ymm0, ymm5, ymm3) // ymm3 = alpha10 * ymm5

	vsubpd(ymm2, ymm6, ymm6) // ymm6 -= ymm2
	vsubpd(ymm3, ymm7, ymm7) // ymm7 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulpd(ymm1, ymm6, ymm6) // ymm6 *= (1/alpha11)
	vmulpd(ymm1, ymm7, ymm7) // ymm7 *= (1/alpha11)
#else
	vdivpd(ymm1, ymm6, ymm6) // ymm6 /= alpha11
	vdivpd(ymm1, ymm7, ymm7) // ymm7 /= alpha11
#endif

	vmovupd(ymm6, mem(rcx)) // store ( beta10..beta13 ) = ymm6
	vmovupd(ymm7, mem(rdx)) // store ( beta14..beta17 ) = ymm7
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 2 -------------

	vbroadcastsd(mem(2+0*6)*8(rax), ymm0) // ymm0 = alpha20
	vbroadcastsd(mem(2+1*6)*8(rax), ymm1) // ymm1 = alpha21

	vmulpd(ymm0, ymm4, ymm2) // ymm2 = alpha20 * ymm4
	vmulpd(ymm0, ymm5, ymm3) // ymm3 = alpha20 * ymm5

	vbroadcastsd(mem(2+2*6)*8(rax), ymm0) // ymm0 = (1/alpha22)

	vfmadd231pd(ymm1, ymm6, ymm2) // ymm2 += alpha21 * ymm6
	vfmadd231pd(ymm1, ymm7, ymm3) // ymm3 += alpha21 * ymm7

	vsubpd(ymm2, ymm8, ymm8) // ymm8 -= ymm2
	vsubpd(ymm3, ymm9, ymm9) // ymm9 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulpd(ymm0, ymm8, ymm8) // ymm8 *= (1/alpha22)
	vmulpd(ymm0, ymm9, ymm9) // ymm9 *= (1/alpha22)
#else
	vdivpd(ymm0, ymm8, ymm8) // ymm8 /= alpha22
	vdivpd(ymm0, ymm9, ymm9) // ymm9 /= alpha22
#endif

	vmovupd(ymm8, mem(rcx)) // store ( beta20..beta23 ) = ymm8
	vmovupd(ymm9, mem(rdx)) // store ( beta24..beta27 ) = ymm9
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 3 -------------

	vbroadcastsd(mem(3+0*6)*8(rax), ymm0) // ymm0 = alpha30
	vbroadcastsd(mem(3+1*6)*8(rax), ymm1) // ymm1 = alpha31

	vmulpd(ymm0, ymm4, ymm2) // ymm2 = alpha30 * ymm4
	vmulpd(ymm0, ymm5, ymm3) // ymm3 = alpha30 * ymm5

	vbroadcastsd(mem(3+2*6)*8(rax), ymm0) // ymm0 = alpha32

	vfmadd231pd(ymm1, ymm6, ymm2) // ymm2 += alpha31 * ymm6
	vfmadd231pd(ymm1, ymm7, ymm3) // ymm3 += alpha31 * ymm7

	vbroadcastsd(mem(3+3*6)*8(rax), ymm1) // ymm1 = (1/alpha33)

	vfmadd231pd(ymm0, ymm8, ymm2) // ymm2 += alpha32 * ymm8
	vfmadd231pd(ymm0, ymm9, ymm3) // ymm3 += alpha32 * ymm9

	vsubpd(ymm2, ymm10, ymm10) // ymm10 -= ymm2
	vsubpd(ymm3, ymm11, ymm11) // ymm11 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulpd(ymm1, ymm10, ymm10) // ymm10 *= (1/alpha33)
	vmulpd(ymm1, ymm11, ymm11) // ymm11 *= (1/alpha33)
#else
	vdivpd(ymm1, ymm10, ymm10) // ymm10 /= alpha33
	vdivpd(ymm1, ymm11, ymm11) // ymm11 /= alpha33
#endif

	vmovupd(ymm10, mem(rcx)) // store ( beta30..beta33 ) = ymm10
	vmovupd(ymm11, mem(rdx)) // store ( beta34..beta37 ) = ymm11
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 4 -------------

	vbroadcastsd(mem(4+0*6)*8(rax), ymm0) // ymm0 = alpha40
	vbroadcastsd(mem(4+1*6)*8(rax), ymm1) // ymm1 = alpha41

	vmulpd(ymm0, ymm4, ymm2) // ymm2 = alpha40 * ymm4
	vmulpd(ymm0, ymm5, ymm3) // ymm3 = alpha40 * ymm5

	vbroadcastsd(mem(4+2*6)*8(rax), ymm0) // ymm0 = alpha42

	vfmadd231pd(ymm1, ymm6, ymm2) // ymm2 += alpha41 * ymm6
	vfmadd231pd(ymm1, ymm7, ymm3) // ymm3 += alpha41 * ymm7

	vbroadcastsd(mem(4+3*6)*8(rax), ymm1) // ymm1 = alpha43

	vfmadd231pd(ymm0, ymm8, ymm2) // ymm2 += alpha42 * ymm8
	vfmadd231pd(ymm0, ymm9, ymm3) // ymm3 += alpha42 * ymm9

	vbroadcastsd(mem(4+4*6)*8(rax), ymm0) // ymm4 = (1/alpha44)

	vfmadd231pd(ymm1, ymm10, ymm2) // ymm2 += alpha43 * ymm10
	vfmadd231pd(ymm1, ymm11, ymm3) // ymm3 += alpha43 * ymm11

	vsubpd(ymm2, ymm12, ymm12) // ymm12 -= ymm2
	vsubpd(ymm3, ymm13, ymm13) // ymm13 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulpd(ymm0, ymm12, ymm12) // ymm12 *= (1/alpha44)
	vmulpd(ymm0, ymm13, ymm13) // ymm13 *= (1/alpha44)
#else
	vdivpd(ymm0, ymm12, ymm12) // ymm12 /= alpha44
	vdivpd(ymm0, ymm13, ymm13) // ymm13 /= alpha44
#endif

	vmovupd(ymm12, mem(rcx)) // store ( beta40..beta43 ) = ymm12
	vmovupd(ymm13, mem(rdx)) // store ( beta44..beta47 ) = ymm13
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b

	 // iteration 5 -------------

	vbroadcastsd(mem(5+0*6)*8(rax), ymm0) // ymm0 = alpha50
	vbroadcastsd(mem(5+1*6)*8(rax), ymm1) // ymm1 = alpha51

	vmulpd(ymm0, ymm4, ymm2) // ymm2 = alpha50 * ymm4
	vmulpd(ymm0, ymm5, ymm3) // ymm3 = alpha50 * ymm5

	vbroadcastsd(mem(5+2*6)*8(rax), ymm0) // ymm0 = alpha52

	vfmadd231pd(ymm1, ymm6, ymm2) // ymm2 += alpha51 * ymm6
	vfmadd231pd(ymm1, ymm7, ymm3) // ymm3 += alpha51 * ymm7

	vbroadcastsd(mem(5+3*6)*8(rax), ymm1) // ymm1 = alpha53

	vfmadd231pd(ymm0, ymm8, ymm2) // ymm2 += alpha52 * ymm8
	vfmadd231pd(ymm0, ymm9, ymm3) // ymm3 += alpha52 * ymm9

	vbroadcastsd(mem(5+4*6)*8(rax), ymm0) // ymm0 = alpha54

	vfmadd231pd(ymm1, ymm10, ymm2) // ymm2 += alpha53 * ymm10
	vfmadd231pd(ymm1, ymm11, ymm3) // ymm3 += alpha53 * ymm11

	vbroadcastsd(mem(5+5*6)*8(rax), ymm1) // ymm1 = (1/alpha55)

	vfmadd231pd(ymm0, ymm12, ymm2) // ymm2 += alpha54 * ymm12
	vfmadd231pd(ymm0, ymm13, ymm3) // ymm3 += alpha54 * ymm13

	vsubpd(ymm2, ymm14, ymm14) // ymm14 -= ymm2
	vsubpd(ymm3, ymm15, ymm15) // ymm15 -= ymm3

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	vmulpd(ymm1, ymm14, ymm14) // ymm14 *= (1/alpha55)
	vmulpd(ymm1, ymm15, ymm15) // ymm15 *= (1/alpha55)
#else
	vdivpd(ymm1, ymm14, ymm14) // ymm14 /= alpha55
	vdivpd(ymm1, ymm15, ymm15) // ymm15 /= alpha55
#endif

	vmovupd(ymm14, mem(rcx)) // store ( beta50..beta53 ) = ymm14
	vmovupd(ymm15, mem(rdx)) // store ( beta54..beta57 ) = ymm15
	add(rdi, rcx) // rcx += rs_b
	add(rdi, rdx) // rdx += rs_b




	mov(r8, rcx) // load address of c11 from r8
	mov(r9, rdi) // load rs_c (in bytes) from r9
	mov(r10, rsi) // load cs_c (in bytes) from r10

	lea(mem(rcx, rsi, 4), rdx) // load address of c11 + 4*cs_c;
	lea(mem(rcx, rdi, 4), r14) // load address of c11 + 4*rs_c;

	 // These are used in the macros below.
	lea(mem(rsi, rsi, 2), r13) // r13 = 3*cs_c;
  //lea(mem(rsi, rsi, 4), r15) // r15 = 5*cs_c;
  //lea(mem(r13, rsi, 4), r10) // r10 = 7*cs_c;



	cmp(imm(8), rsi) // set ZF if (8*cs_c) == 8.
	jz(.DROWSTORED) // jump to row storage case



	cmp(imm(8), rdi) // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED) // jump to column storage case



	 // if neither row- or column-
	 // stored, use general case.
	label(.DGENSTORED)


	vmovapd(ymm4, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm6, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm8, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm10, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm12, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm14, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ


	mov(rdx, rcx) // rcx = c11 + 4*cs_c


	vmovapd(ymm5, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm7, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm9, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm11, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm13, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ
	add(rdi, rcx) // c11 += rs_c;


	vmovapd(ymm15, ymm0)
	DGEMM_OUTPUT_GS_BETA_NZ


	jmp(.DDONE)



	label(.DROWSTORED)


	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	vmovupd(ymm5, mem(rdx))
	add(rdi, rdx)

	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)
	vmovupd(ymm7, mem(rdx))
	add(rdi, rdx)

	vmovupd(ymm8, mem(rcx))
	add(rdi, rcx)
	vmovupd(ymm9, mem(rdx))
	add(rdi, rdx)

	vmovupd(ymm10, mem(rcx))
	add(rdi, rcx)
	vmovupd(ymm11, mem(rdx))
	add(rdi, rdx)

	vmovupd(ymm12, mem(rcx))
	add(rdi, rcx)
	vmovupd(ymm13, mem(rdx))
	add(rdi, rdx)

	vmovupd(ymm14, mem(rcx))
	//add(rdi, rcx)
	vmovupd(ymm15, mem(rdx))
	//add(rdi, rdx)


	jmp(.DDONE)



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, r13, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovupd(xmm0, mem(r14))
	vmovupd(xmm1, mem(r14, rsi, 1))
	vmovupd(xmm2, mem(r14, rsi, 2))
	vmovupd(xmm3, mem(r14, r13, 1))

	lea(mem(r14, rsi, 4), r14)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, r13, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm3)

	vmovupd(xmm0, mem(r14))
	vmovupd(xmm1, mem(r14, rsi, 1))
	vmovupd(xmm2, mem(r14, rsi, 2))
	vmovupd(xmm3, mem(r14, r13, 1))

	//lea(mem(r14, rsi, 4), r14)




	label(.DDONE)

	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	  [k_iter] "m" (k_iter), // 0
	  [k_left] "m" (k_left), // 1
	  [a10]    "m" (a10),    // 2
	  [b01]    "m" (b01),    // 3
	  [beta]   "m" (beta),   // 4
	  [alpha]  "m" (alpha),  // 5
	  [a11]    "m" (a11),    // 6
	  [b11]    "m" (b11),    // 7
	  [c11]    "m" (c11),    // 8
	  [rs_c]   "m" (rs_c),   // 9
	  [cs_c]   "m" (cs_c)    // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMMTRSM_UKR_FLUSH_CT( d );
}



