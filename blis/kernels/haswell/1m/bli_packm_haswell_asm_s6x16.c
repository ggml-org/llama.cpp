/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2020, Advanced Micro Devices, Inc.

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

void bli_spackm_haswell_asm_6x16
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim0,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   k0,
             dim_t   k0_max,
       const void*   kappa,
       const void*   a, inc_t inca0, inc_t lda0,
             void*   p,              inc_t ldp0,
       const void*   params,
       const cntx_t* cntx
     )
{
	// This is the panel dimension assumed by the packm kernel.
	const dim_t      mr    = 6;
	const dim_t      nr    = 16;

	// This is the "packing" dimension assumed by the packm kernel.
	// This should be equal to ldp.
	//const dim_t    packmnr = 8;

	// Define a local copy of 1.0 so we can test for unit kappa.
	float            one_l = 1.0;
	float*  restrict one   = &one_l;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	const uint64_t k_iter = k0 / 8;
#if 1
	const uint64_t k_left = k0 % 8;
#else
	const uint64_t k_left = k0;
#endif

	// NOTE: For the purposes of the comments in this packm kernel, we
	// interpret inca and lda as rs_a and cs_a, respectively, and similarly
	// interpret ldp as cs_p (with rs_p implicitly unit). Thus, when reading
	// this packm kernel, you should think of the operation as packing an
	// m x n micropanel, where m and n are tiny and large, respectively, and
	// where elements of each column of the packed matrix P are contiguous.
	// (This packm kernel can still be used to pack micropanels of matrix B
	// in a gemm operation.)
	const uint64_t inca   = inca0;
	const uint64_t lda    = lda0;
	const uint64_t ldp    = ldp0;

	const bool     gs     = ( inca0 != 1 && lda0 != 1 );

	// NOTE: If/when this kernel ever supports scaling by kappa within the
	// assembly region, this constraint should be lifted.
	const bool     unitk  = bli_seq1( *(( float* )kappa) );


	// -------------------------------------------------------------------------

	if ( cdim0 == mr && ldp0 == mr && cdim_bcast == 1 && !gs && unitk )
	{
		begin_asm()

		mov(var(a), rax)                   // load address of a.

		mov(var(inca), r8)                 // load inca
		mov(var(lda), r10)                 // load lda
		lea(mem(, r8,  4), r8)             // inca *= sizeof(float)
		lea(mem(, r10, 4), r10)            // lda *= sizeof(float)

		mov(var(p), rbx)                   // load address of p.

		lea(mem(   , r10, 8), r14)         // r14 = 8*lda

		mov(var(one), rdx)                 // load address of 1.0 constant
		vmovss(mem(rdx), xmm1)             // load 1.0

		mov(var(kappa), rcx)               // load address of kappa
		vmovss(mem(rcx), xmm0)             // load kappa


										   // now branch on kappa == 1.0

		vucomiss(xmm0, xmm1)               // set ZF if kappa == 1.0
		je(.SKAPPAUNIT)                    // if ZF = 1, jump to beta == 0 case



		label(.SKAPPANONU)

		cmp(imm(4), r8)                    // set ZF if (4*inca) == 4.
		jz(.SCOLNONU)                      // jump to column storage case

		// -- kappa non-unit, row storage on A -------------------------------------

		label(.SROWNONU)

		jmp(.SDONE)                        // jump to end.


		// -- kappa non-unit, column storage on A ----------------------------------

		label(.SCOLNONU)

		jmp(.SDONE)                        // jump to end.




		label(.SKAPPAUNIT)

		cmp(imm(4), r8)                    // set ZF if (4*inca) == 4.
		jz(.SCOLUNIT)                      // jump to column storage case


		// -- kappa unit, row storage on A -----------------------------------------

		label(.SROWUNIT)

		lea(mem(r8,  r8,  2), r13)         // r13 = 3*inca
		lea(mem(r13, r8,  2), r15)         // r15 = 5*inca
		//lea(mem(r13, r8,  4), rdx)         // rdx = 7*inca

		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SCONKLEFTROWU)                 // if i == 0, jump to code that
		                                   // contains the k_left loop.


		label(.SKITERROWU)                 // MAIN LOOP (k_iter)

		                                   // begin IO on rows 0-3
		vmovups(mem(rax,         0), ymm4)
		vmovups(mem(rax,  r8, 1, 0), ymm6)
		vmovups(mem(rax,  r8, 2, 0), ymm8)
		vmovups(mem(rax, r13, 1, 0), ymm10)

		vunpcklps(ymm6, ymm4, ymm0)
		vunpcklps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rbx, 0*24))      // store ( gamma00..gamma30 )
		vmovups(xmm2, mem(rbx, 4*24))      // store ( gamma04..gamma34 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rbx, 1*24))      // store ( gamma01..gamma31 )
		vmovups(xmm2, mem(rbx, 5*24))      // store ( gamma05..gamma35 )

		vunpckhps(ymm6, ymm4, ymm0)
		vunpckhps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rbx, 2*24))      // store ( gamma02..gamma32 )
		vmovups(xmm2, mem(rbx, 6*24))      // store ( gamma06..gamma36 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rbx, 3*24))      // store ( gamma03..gamma33 )
		vmovups(xmm2, mem(rbx, 7*24))      // store ( gamma07..gamma37 )

		                                   // begin IO on rows 4-5
		vmovups(mem(rax,  r8, 4, 0), ymm12)
		vmovups(mem(rax, r15, 1, 0), ymm14)

		vunpcklps(ymm14, ymm12, ymm0)
		vextractf128(imm(0x1), ymm0, xmm2)
		vmovlpd(xmm0, mem(rbx, 0*24+16))   // store ( gamma40..gamma50 )
		vmovhpd(xmm0, mem(rbx, 1*24+16))   // store ( gamma41..gamma51 )
		vmovlpd(xmm2, mem(rbx, 4*24+16))   // store ( gamma44..gamma54 )
		vmovhpd(xmm2, mem(rbx, 5*24+16))   // store ( gamma45..gamma55 )

		vunpckhps(ymm14, ymm12, ymm0)
		vextractf128(imm(0x1), ymm0, xmm2)
		vmovlpd(xmm0, mem(rbx, 2*24+16))   // store ( gamma42..gamma52 )
		vmovhpd(xmm0, mem(rbx, 3*24+16))   // store ( gamma43..gamma53 )
		vmovlpd(xmm2, mem(rbx, 6*24+16))   // store ( gamma46..gamma56 )
		vmovhpd(xmm2, mem(rbx, 7*24+16))   // store ( gamma47..gamma57 )


		add(r14, rax)                      // a += 8*lda;
		add(imm(8*6*4), rbx)               // p += 8*ldp = 8*6;

		dec(rsi)                           // i -= 1;
		jne(.SKITERROWU)                   // iterate again if i != 0.



		label(.SCONKLEFTROWU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.


		label(.SKLEFTROWU)                 // EDGE LOOP (k_left)

		vmovss(mem(rax,         0), xmm0)
		vmovss(mem(rax,  r8, 1, 0), xmm2)
		vmovss(mem(rax,  r8, 2, 0), xmm4)
		vmovss(mem(rax, r13, 1, 0), xmm6)
		vmovss(mem(rax,  r8, 4, 0), xmm1)
		vmovss(mem(rax, r15, 1, 0), xmm3)

		vmovss(xmm0, mem(rbx, 0*4))
		vmovss(xmm2, mem(rbx, 1*4))
		vmovss(xmm4, mem(rbx, 2*4))
		vmovss(xmm6, mem(rbx, 3*4))
		vmovss(xmm1, mem(rbx, 4*4))
		vmovss(xmm3, mem(rbx, 5*4))

		add(r10, rax)                      // a += lda;
		add(imm(6*4), rbx)                 // p += ldp = 6;

		dec(rsi)                           // i -= 1;
		jne(.SKLEFTROWU)                   // iterate again if i != 0.


		jmp(.SDONE)                        // jump to end.


		// -- kappa unit, column storage on A --------------------------------------

		label(.SCOLUNIT)

		lea(mem(r10, r10, 2), r13)         // r13 = 3*lda
		lea(mem(r13, r10, 2), r15)         // r15 = 5*lda
		lea(mem(r13, r10, 4), rdx)         // rdx = 7*lda

		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SCONKLEFTCOLU)                 // if i == 0, jump to code that
		                                   // contains the k_left loop.


		label(.SKITERCOLU)                 // MAIN LOOP (k_iter)

		vmovups(mem(rax,          0), xmm0)
		vmovsd( mem(rax,         16), xmm1)
		vmovups(xmm0, mem(rbx, 0*24+ 0))
		vmovsd( xmm1, mem(rbx, 0*24+16))

		vmovups(mem(rax, r10, 1,  0), xmm2)
		vmovsd( mem(rax, r10, 1, 16), xmm3)
		vmovups(xmm2, mem(rbx, 1*24+ 0))
		vmovsd( xmm3, mem(rbx, 1*24+16))

		vmovups(mem(rax, r10, 2,  0), xmm4)
		vmovsd( mem(rax, r10, 2, 16), xmm5)
		vmovups(xmm4, mem(rbx, 2*24+ 0))
		vmovsd( xmm5, mem(rbx, 2*24+16))

		vmovups(mem(rax, r13, 1,  0), xmm6)
		vmovsd( mem(rax, r13, 1, 16), xmm7)
		vmovups(xmm6, mem(rbx, 3*24+ 0))
		vmovsd( xmm7, mem(rbx, 3*24+16))

		vmovups(mem(rax, r10, 4,  0), xmm8)
		vmovsd( mem(rax, r10, 4, 16), xmm9)
		vmovups(xmm8, mem(rbx, 4*24+ 0))
		vmovsd( xmm9, mem(rbx, 4*24+16))

		vmovups(mem(rax, r15, 1,  0), xmm10)
		vmovsd( mem(rax, r15, 1, 16), xmm11)
		vmovups(xmm10, mem(rbx, 5*24+ 0))
		vmovsd( xmm11, mem(rbx, 5*24+16))

		vmovups(mem(rax, r13, 2,  0), xmm12)
		vmovsd( mem(rax, r13, 2, 16), xmm13)
		vmovups(xmm12, mem(rbx, 6*24+ 0))
		vmovsd( xmm13, mem(rbx, 6*24+16))

		vmovups(mem(rax, rdx, 1,  0), xmm14)
		vmovsd( mem(rax, rdx, 1, 16), xmm15)
		vmovups(xmm14, mem(rbx, 7*24+ 0))
		vmovsd( xmm15, mem(rbx, 7*24+16))

		add(r14, rax)                      // a += 8*lda;
		add(imm(8*6*4), rbx)               // p += 8*ldp = 8*6;

		dec(rsi)                           // i -= 1;
		jne(.SKITERCOLU)                   // iterate again if i != 0.



		label(.SCONKLEFTCOLU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.


		label(.SKLEFTCOLU)                 // EDGE LOOP (k_left)

		vmovups(mem(rax,          0), xmm0)
		vmovsd( mem(rax,         16), xmm1)
		add(r10, rax)                      // a += lda;
		vmovups(xmm0, mem(rbx, 0*24+ 0))
		vmovsd( xmm1, mem(rbx, 0*24+16))
		add(imm(6*4), rbx)                 // p += ldp = 6;

		dec(rsi)                           // i -= 1;
		jne(.SKLEFTCOLU)                   // iterate again if i != 0.


		//jmp(.SDONE)                        // jump to end.



		label(.SDONE)



		end_asm(
		: // output operands (none)
		: // input operands
		  [k_iter] "m" (k_iter),
		  [k_left] "m" (k_left),
		  [a]      "m" (a),
		  [inca]   "m" (inca),
		  [lda]    "m" (lda),
		  [p]      "m" (p),
		  [ldp]    "m" (ldp),
		  [kappa]  "m" (kappa),
		  [one]    "m" (one)
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "r8", /*"r9",*/ "r10", /*"r11",*/ "r12", "r13", "r14", "r15",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
		)
	}
	else if ( cdim0 == nr && ldp0 == nr && cdim_bcast == 1 && !gs && unitk )
	{
		begin_asm()

		mov(var(a), rax)                   // load address of a.

		mov(var(inca), r8)                 // load inca
		mov(var(lda), r10)                 // load lda
		lea(mem(, r8,  4), r8)             // inca *= sizeof(float)
		lea(mem(, r10, 4), r10)            // lda *= sizeof(float)

		mov(var(p), rbx)                   // load address of p.

		lea(mem(   , r10, 8), r14)         // r14 = 8*lda

		mov(var(one), rdx)                 // load address of 1.0 constant
		vmovss(mem(rdx), xmm1)             // load 1.0

		mov(var(kappa), rcx)               // load address of kappa
		vmovss(mem(rcx), xmm0)             // load kappa


										   // now branch on kappa == 1.0

		vucomiss(xmm0, xmm1)               // set ZF if kappa == 1.0
		je(.SKAPPAUNIT)                    // if ZF = 1, jump to beta == 0 case



		label(.SKAPPANONU)

		cmp(imm(4), r8)                    // set ZF if (4*inca) == 4.
		jz(.SCOLNONU)                      // jump to column storage case

		// -- kappa non-unit, row storage on A -------------------------------------

		label(.SROWNONU)

		jmp(.SDONE)                        // jump to end.


		// -- kappa non-unit, column storage on A ----------------------------------

		label(.SCOLNONU)

		jmp(.SDONE)                        // jump to end.




		label(.SKAPPAUNIT)

		cmp(imm(4), r8)                    // set ZF if (4*inca) == 4.
		jz(.SCOLUNIT)                      // jump to column storage case


		// -- kappa unit, row storage on A -----------------------------------------

		label(.SROWUNIT)

		lea(mem(r8,  r8,  2), r13)         // r13 = 3*inca
		lea(mem(r13, r8,  2), r15)         // r15 = 5*inca
		lea(mem(r13, r8,  4), rdx)         // rdx = 7*inca

		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SCONKLEFTROWU)                 // if i == 0, jump to code that
		                                   // contains the k_left loop.


		label(.SKITERROWU)                 // MAIN LOOP (k_iter)

		mov(rax, r12)                      // r12 = rax
		mov(rbx, rcx)                      // rcx = rbx


		                                   // begin IO on rows 0-3
		vmovups(mem(r12,         0), ymm4)
		vmovups(mem(r12,  r8, 1, 0), ymm6)
		vmovups(mem(r12,  r8, 2, 0), ymm8)
		vmovups(mem(r12, r13, 1, 0), ymm10)

		vunpcklps(ymm6, ymm4, ymm0)
		vunpcklps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 0*64))      // store ( gamma00..gamma30 )
		vmovups(xmm2, mem(rcx, 4*64))      // store ( gamma04..gamma34 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 1*64))      // store ( gamma01..gamma31 )
		vmovups(xmm2, mem(rcx, 5*64))      // store ( gamma05..gamma35 )

		vunpckhps(ymm6, ymm4, ymm0)
		vunpckhps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 2*64))      // store ( gamma02..gamma32 )
		vmovups(xmm2, mem(rcx, 6*64))      // store ( gamma06..gamma36 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 3*64))      // store ( gamma03..gamma33 )
		vmovups(xmm2, mem(rcx, 7*64))      // store ( gamma07..gamma37 )


		lea(mem(r12, r8, 4), r12)          // r12 += 4*inca
		add(imm(4*4), rcx)                 // rcx += 4;


		                                   // begin IO on rows 4-7
		vmovups(mem(r12,         0), ymm4)
		vmovups(mem(r12,  r8, 1, 0), ymm6)
		vmovups(mem(r12,  r8, 2, 0), ymm8)
		vmovups(mem(r12, r13, 1, 0), ymm10)

		vunpcklps(ymm6, ymm4, ymm0)
		vunpcklps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 0*64))      // store ( gamma40..gamma70 )
		vmovups(xmm2, mem(rcx, 4*64))      // store ( gamma44..gamma74 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 1*64))      // store ( gamma41..gamma71 )
		vmovups(xmm2, mem(rcx, 5*64))      // store ( gamma45..gamma75 )

		vunpckhps(ymm6, ymm4, ymm0)
		vunpckhps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 2*64))      // store ( gamma42..gamma72 )
		vmovups(xmm2, mem(rcx, 6*64))      // store ( gamma46..gamma76 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 3*64))      // store ( gamma43..gamma73 )
		vmovups(xmm2, mem(rcx, 7*64))      // store ( gamma47..gamma77 )


		lea(mem(r12, r8, 4), r12)          // r12 += 4*inca
		add(imm(4*4), rcx)                 // rcx += 4;


		                                   // begin IO on rows 8-11
		vmovups(mem(r12,         0), ymm4)
		vmovups(mem(r12,  r8, 1, 0), ymm6)
		vmovups(mem(r12,  r8, 2, 0), ymm8)
		vmovups(mem(r12, r13, 1, 0), ymm10)

		vunpcklps(ymm6, ymm4, ymm0)
		vunpcklps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 0*64))      // store ( gamma80..gammaB0 )
		vmovups(xmm2, mem(rcx, 4*64))      // store ( gamma84..gammaB4 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 1*64))      // store ( gamma81..gammaB1 )
		vmovups(xmm2, mem(rcx, 5*64))      // store ( gamma85..gammaB5 )

		vunpckhps(ymm6, ymm4, ymm0)
		vunpckhps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 2*64))      // store ( gamma82..gammaB2 )
		vmovups(xmm2, mem(rcx, 6*64))      // store ( gamma86..gammaB6 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 3*64))      // store ( gamma83..gammaB3 )
		vmovups(xmm2, mem(rcx, 7*64))      // store ( gamma87..gammaB7 )


		lea(mem(r12, r8, 4), r12)          // r12 += 4*inca
		add(imm(4*4), rcx)                 // rcx += 4;


		                                   // begin IO on rows 12-15
		vmovups(mem(r12,         0), ymm4)
		vmovups(mem(r12,  r8, 1, 0), ymm6)
		vmovups(mem(r12,  r8, 2, 0), ymm8)
		vmovups(mem(r12, r13, 1, 0), ymm10)

		vunpcklps(ymm6, ymm4, ymm0)
		vunpcklps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 0*64))      // store ( gammaC0..gammaF0 )
		vmovups(xmm2, mem(rcx, 4*64))      // store ( gammaC4..gammaF4 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 1*64))      // store ( gammaC1..gammaF1 )
		vmovups(xmm2, mem(rcx, 5*64))      // store ( gammaC5..gammaF5 )

		vunpckhps(ymm6, ymm4, ymm0)
		vunpckhps(ymm10, ymm8, ymm1)
		vshufps(imm(0x4e), ymm1, ymm0, ymm2)
		vblendps(imm(0xcc), ymm2, ymm0, ymm0)
		vblendps(imm(0x33), ymm2, ymm1, ymm1)

		vextractf128(imm(0x1), ymm0, xmm2)
		vmovups(xmm0, mem(rcx, 2*64))      // store ( gammaC2..gammaF2 )
		vmovups(xmm2, mem(rcx, 6*64))      // store ( gammaC6..gammaF6 )

		vextractf128(imm(0x1), ymm1, xmm2)
		vmovups(xmm1, mem(rcx, 3*64))      // store ( gammaC3..gammaF3 )
		vmovups(xmm2, mem(rcx, 7*64))      // store ( gammaC7..gammaF7 )


		add(r14, rax)                      // a += 8*lda;
		add(imm(8*16*4), rbx)              // p += 8*ldp = 8*16;

		dec(rsi)                           // i -= 1;
		jne(.SKITERROWU)                   // iterate again if i != 0.



		label(.SCONKLEFTROWU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.


		label(.SKLEFTROWU)                 // EDGE LOOP (k_left)

		vmovss(mem(rax,         0), xmm0)
		vmovss(mem(rax,  r8, 1, 0), xmm2)
		vmovss(mem(rax,  r8, 2, 0), xmm4)
		vmovss(mem(rax, r13, 1, 0), xmm6)
		vmovss(mem(rax,  r8, 4, 0), xmm1)
		vmovss(mem(rax, r15, 1, 0), xmm3)
		vmovss(mem(rax, r13, 2, 0), xmm5)
		vmovss(mem(rax, rdx, 1, 0), xmm7)

		vmovss(xmm0, mem(rbx, 0*4))
		vmovss(xmm2, mem(rbx, 1*4))
		vmovss(xmm4, mem(rbx, 2*4))
		vmovss(xmm6, mem(rbx, 3*4))
		vmovss(xmm1, mem(rbx, 4*4))
		vmovss(xmm3, mem(rbx, 5*4))
		vmovss(xmm5, mem(rbx, 6*4))
		vmovss(xmm7, mem(rbx, 7*4))

		lea(mem(rax, r8, 8), r12)         // r12 = a + 8*inca

		vmovss(mem(r12,         0), xmm0)
		vmovss(mem(r12,  r8, 1, 0), xmm2)
		vmovss(mem(r12,  r8, 2, 0), xmm4)
		vmovss(mem(r12, r13, 1, 0), xmm6)
		vmovss(mem(r12,  r8, 4, 0), xmm1)
		vmovss(mem(r12, r15, 1, 0), xmm3)
		vmovss(mem(r12, r13, 2, 0), xmm5)
		vmovss(mem(r12, rdx, 1, 0), xmm7)

		add(r10, rax)                      // a += lda;

		vmovss(xmm0, mem(rbx,  8*4))
		vmovss(xmm2, mem(rbx,  9*4))
		vmovss(xmm4, mem(rbx, 10*4))
		vmovss(xmm6, mem(rbx, 11*4))
		vmovss(xmm1, mem(rbx, 12*4))
		vmovss(xmm3, mem(rbx, 13*4))
		vmovss(xmm5, mem(rbx, 14*4))
		vmovss(xmm7, mem(rbx, 15*4))

		add(imm(16*4), rbx)                // p += ldp = 16;

		dec(rsi)                           // i -= 1;
		jne(.SKLEFTROWU)                   // iterate again if i != 0.


		jmp(.SDONE)                        // jump to end.


		// -- kappa unit, column storage on A --------------------------------------

		label(.SCOLUNIT)

		lea(mem(r10, r10, 2), r13)         // r13 = 3*lda
		lea(mem(r13, r10, 2), r15)         // r15 = 5*lda
		lea(mem(r13, r10, 4), rdx)         // rdx = 7*lda

		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SCONKLEFTCOLU)                 // if i == 0, jump to code that
		                                   // contains the k_left loop.


		label(.SKITERCOLU)                 // MAIN LOOP (k_iter)

		vmovups(mem(rax,          0), ymm0)
		vmovups(mem(rax,         32), ymm1)
		vmovups(ymm0, mem(rbx, 0*64+ 0))
		vmovups(ymm1, mem(rbx, 0*64+32))

		vmovups(mem(rax, r10, 1,  0), ymm2)
		vmovups(mem(rax, r10, 1, 32), ymm3)
		vmovups(ymm2, mem(rbx, 1*64+ 0))
		vmovups(ymm3, mem(rbx, 1*64+32))

		vmovups(mem(rax, r10, 2,  0), ymm4)
		vmovups(mem(rax, r10, 2, 32), ymm5)
		vmovups(ymm4, mem(rbx, 2*64+ 0))
		vmovups(ymm5, mem(rbx, 2*64+32))

		vmovups(mem(rax, r13, 1,  0), ymm6)
		vmovups(mem(rax, r13, 1, 32), ymm7)
		vmovups(ymm6, mem(rbx, 3*64+ 0))
		vmovups(ymm7, mem(rbx, 3*64+32))

		vmovups(mem(rax, r10, 4,  0), ymm8)
		vmovups(mem(rax, r10, 4, 32), ymm9)
		vmovups(ymm8, mem(rbx, 4*64+ 0))
		vmovups(ymm9, mem(rbx, 4*64+32))

		vmovups(mem(rax, r15, 1,  0), ymm10)
		vmovups(mem(rax, r15, 1, 32), ymm11)
		vmovups(ymm10, mem(rbx, 5*64+ 0))
		vmovups(ymm11, mem(rbx, 5*64+32))

		vmovups(mem(rax, r13, 2,  0), ymm12)
		vmovups(mem(rax, r13, 2, 32), ymm13)
		vmovups(ymm12, mem(rbx, 6*64+ 0))
		vmovups(ymm13, mem(rbx, 6*64+32))

		vmovups(mem(rax, rdx, 1,  0), ymm14)
		vmovups(mem(rax, rdx, 1, 32), ymm15)
		add(r14, rax)                      // a += 8*lda;
		vmovups(ymm14, mem(rbx, 7*64+ 0))
		vmovups(ymm15, mem(rbx, 7*64+32))
		add(imm(8*16*4), rbx)              // p += 8*ldp = 8*16;

		dec(rsi)                           // i -= 1;
		jne(.SKITERCOLU)                   // iterate again if i != 0.



		label(.SCONKLEFTCOLU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.SDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.


		label(.SKLEFTCOLU)                 // EDGE LOOP (k_left)

		vmovups(mem(rax,          0), ymm0)
		vmovups(mem(rax,         32), ymm1)
		add(r10, rax)                      // a += lda;
		vmovups(ymm0, mem(rbx, 0*64+ 0))
		vmovups(ymm1, mem(rbx, 0*64+32))
		add(imm(16*4), rbx)                // p += ldp = 16;

		dec(rsi)                           // i -= 1;
		jne(.SKLEFTCOLU)                   // iterate again if i != 0.


		//jmp(.SDONE)                        // jump to end.



		label(.SDONE)



		end_asm(
		: // output operands (none)
		: // input operands
		  [k_iter] "m" (k_iter),
		  [k_left] "m" (k_left),
		  [a]      "m" (a),
		  [inca]   "m" (inca),
		  [lda]    "m" (lda),
		  [p]      "m" (p),
		  [ldp]    "m" (ldp),
		  [kappa]  "m" (kappa),
		  [one]    "m" (one)
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "r8", /*"r9",*/ "r10", /*"r11",*/ "r12", "r13", "r14", "r15",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
		)
	}
	else
	{
		bli_sscal2bbs_mxn
		(
		  conja,
		  cdim0,
		  k0,
		  kappa,
		  a,       inca, lda,
		  p, cdim_bcast, ldp
		);
	}

	bli_sset0s_edge
	(
	  cdim0*cdim_bcast, cdim_max*cdim_bcast,
	  k0, k0_max,
	  p, ldp
	);
}

