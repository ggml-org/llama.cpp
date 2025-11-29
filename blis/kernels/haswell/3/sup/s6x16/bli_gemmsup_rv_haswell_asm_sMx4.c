/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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
GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref )


void bli_sgemmsup_rv_haswell_asm_6x4
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
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rbp)         // rbp = 3*cs_c;
	prefetch(0, mem(rcx,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rcx, rbp, 1, 5*8)) // prefetch c + 3*cs_c

	label(.SPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	vbroadcastss(mem(rax, r15, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)
	vfmadd231ps(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	vbroadcastss(mem(rax, r15, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)
	vfmadd231ps(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	vbroadcastss(mem(rax, r15, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)
	vfmadd231ps(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	vbroadcastss(mem(rax, r15, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)
	vfmadd231ps(xmm0, xmm3, xmm14)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	vbroadcastss(mem(rax, r15, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)
	vfmadd231ps(xmm0, xmm3, xmm14)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

	vmulps(xmm0, xmm4, xmm4)           // scale by alpha
	vmulps(xmm0, xmm6, xmm6)
	vmulps(xmm0, xmm8, xmm8)
	vmulps(xmm0, xmm10, xmm10)
	vmulps(xmm0, xmm12, xmm12)
	vmulps(xmm0, xmm14, xmm14)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm4)
	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm6)
	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm8)
	vmovups(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm10)
	vmovups(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm12)
	vmovups(xmm12, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm14)
	vmovups(xmm14, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vunpcklps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )

	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )

	vunpckhps(xmm6, xmm4, xmm0)
	vunpckhps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )

	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vunpcklps(xmm14, xmm12, xmm0)
	vmovlpd(mem(rdx        ), xmm1, xmm1)
	vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx        ))    // store ( gamma40..gamma50 )
	vmovhpd(xmm0, mem(rdx, rsi, 1))    // store ( gamma41..gamma51 )

	vunpckhps(xmm14, xmm12, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx, rsi, 2))    // store ( gamma42..gamma52 )
	vmovhpd(xmm0, mem(rdx, rax, 1))    // store ( gamma43..gamma53 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm12, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm14, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vunpcklps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )

	vunpckhps(xmm6, xmm4, xmm0)
	vunpckhps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vunpcklps(xmm14, xmm12, xmm0)
	vmovlpd(xmm0, mem(rdx        ))    // store ( gamma40..gamma50 )
	vmovhpd(xmm0, mem(rdx, rsi, 1))    // store ( gamma41..gamma51 )

	vunpckhps(xmm14, xmm12, xmm0)
	vmovlpd(xmm0, mem(rdx, rsi, 2))    // store ( gamma42..gamma52 )
	vmovhpd(xmm0, mem(rdx, rax, 1))    // store ( gamma43..gamma53 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c




	label(.SDONE)



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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)
}

void bli_sgemmsup_rv_haswell_asm_5x4
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
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rbp)         // rbp = 3*cs_c;
	prefetch(0, mem(rcx,         4*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rcx, rbp, 1, 4*8)) // prefetch c + 3*cs_c

	label(.SPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)

	vbroadcastss(mem(rax, r8,  4), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm12)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

	vmulps(xmm0, xmm4, xmm4)           // scale by alpha
	vmulps(xmm0, xmm6, xmm6)
	vmulps(xmm0, xmm8, xmm8)
	vmulps(xmm0, xmm10, xmm10)
	vmulps(xmm0, xmm12, xmm12)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm4)
	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm6)
	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm8)
	vmovups(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm10)
	vmovups(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm12)
	vmovups(xmm12, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vunpcklps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )

	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )

	vunpckhps(xmm6, xmm4, xmm0)
	vunpckhps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )

	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(xmm12, xmm0)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(mem(rdx        ), xmm1)
	vmovss(mem(rdx, rsi, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(mem(rdx, rsi, 2), xmm1)
	vmovss(mem(rdx, rax, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma43 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm12, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vunpcklps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )


	vunpckhps(xmm6, xmm4, xmm0)
	vunpckhps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(xmm12, xmm0)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma43 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c




	label(.SDONE)



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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)
}

void bli_sgemmsup_rv_haswell_asm_4x4
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
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rbp)         // rbp = 3*cs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rcx, rbp, 1, 3*8)) // prefetch c + 3*cs_c

	label(.SPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	vbroadcastss(mem(rax, r13, 1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)
	vfmadd231ps(xmm0, xmm3, xmm10)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

	vmulps(xmm0, xmm4, xmm4)           // scale by alpha
	vmulps(xmm0, xmm6, xmm6)
	vmulps(xmm0, xmm8, xmm8)
	vmulps(xmm0, xmm10, xmm10)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm4)
	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm6)
	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm8)
	vmovups(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm10)
	vmovups(xmm10, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vunpcklps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )

	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )

	vunpckhps(xmm6, xmm4, xmm0)
	vunpckhps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )

	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm10, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vunpcklps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )


	vunpckhps(xmm6, xmm4, xmm0)
	vunpckhps(xmm10, xmm8, xmm1)
	vshufps(imm(0x4e), xmm1, xmm0, xmm2)
	vblendps(imm(0xcc), xmm2, xmm0, xmm0)
	vblendps(imm(0x33), xmm2, xmm1, xmm1)

	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c




	label(.SDONE)



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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)
}

void bli_sgemmsup_rv_haswell_asm_3x4
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
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rbp)         // rbp = 3*cs_c;
	prefetch(0, mem(rcx,         2*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rcx, rbp, 1, 2*8)) // prefetch c + 3*cs_c

	label(.SPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)

	vbroadcastss(mem(rax, r8,  2), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm8)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

	vmulps(xmm0, xmm4, xmm4)           // scale by alpha
	vmulps(xmm0, xmm6, xmm6)
	vmulps(xmm0, xmm8, xmm8)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm4)
	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm6)
	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm8)
	vmovups(xmm8, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vmovlpd(mem(rcx        ), xmm1, xmm1)
	vmovhpd(mem(rcx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )

	vunpckhps(xmm6, xmm4, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(xmm8, xmm0)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(mem(rdx        ), xmm1)
	vmovss(mem(rdx, rsi, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma20 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma21 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(mem(rdx, rsi, 2), xmm1)
	vmovss(mem(rdx, rax, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma22 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma23 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm8, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )

	vunpckhps(xmm6, xmm4, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(xmm8, xmm0)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma20 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma21 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma22 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma23 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c




	label(.SDONE)



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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)
}

void bli_sgemmsup_rv_haswell_asm_2x4
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
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rbp)         // rbp = 3*cs_c;
	prefetch(0, mem(rcx,         1*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rcx, rbp, 1, 1*8)) // prefetch c + 3*cs_c

	label(.SPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	vbroadcastss(mem(rax, r8,  1), xmm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)
	vfmadd231ps(xmm0, xmm3, xmm6)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

	vmulps(xmm0, xmm4, xmm4)           // scale by alpha
	vmulps(xmm0, xmm6, xmm6)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 2), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm4)
	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm6)
	vmovups(xmm6, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vmovlpd(mem(rcx        ), xmm1, xmm1)
	vmovhpd(mem(rcx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )

	vunpckhps(xmm6, xmm4, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovups(xmm6, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-3
	vunpcklps(xmm6, xmm4, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )

	vunpckhps(xmm6, xmm4, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c




	label(.SDONE)



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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)
}

void bli_sgemmsup_rv_haswell_asm_1x4
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
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rbp)         // rbp = 3*cs_c;
	prefetch(0, mem(rcx,         0*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rcx, rbp, 1, 0*8)) // prefetch c + 3*cs_c

	label(.SPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)


	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 0
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovups(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), xmm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(xmm0, xmm2, xmm4)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

	vmulps(xmm0, xmm4, xmm4)           // scale by alpha






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 2), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), xmm3, xmm4)
	vmovups(xmm4, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)


	                                   // begin I/O on columns 0-3
	vmovups(xmm4, xmm0)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(mem(rcx        ), xmm1)
	vmovss(mem(rcx, rsi, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx        ))     // store ( gamma00 )
	vmovss(xmm4, mem(rcx, rsi, 1))     // store ( gamma01 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(mem(rcx, rsi, 2), xmm1)
	vmovss(mem(rcx, rax, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 2))     // store ( gamma02 )
	vmovss(xmm4, mem(rcx, rax, 1))     // store ( gamma03 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(xmm4, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-3
	vmovups(xmm4, xmm0)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rcx        ))     // store ( gamma00 )
	vmovss(xmm4, mem(rcx, rsi, 1))     // store ( gamma01 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 2))     // store ( gamma02 )
	vmovss(xmm4, mem(rcx, rax, 1))     // store ( gamma03 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c




	label(.SDONE)



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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)
}

