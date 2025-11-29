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

// This avoids a known issue with GCC15+ ("error: bp cannot be used in asm here", #845).
// Make sure the compiler isn't clang since it also confusingly defines __GNUC__
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 15
#pragma GCC optimize("-fno-tree-vectorize")
#endif

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


void bli_sgemmsup_rv_haswell_asm_6x16n
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
	uint64_t m_left = m0 % 6;

	// First check whether this is a edge case in the m dimension. If so,
	// dispatch other ?x8m kernels, as needed.
	if ( m_left )
	{
		      float* cij = ( float* )c;
		const float* bj  = ( float* )b;
		const float* ai  = ( float* )a;

#if 1
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m0 )
		{
			gemmsup_ker_ft  ker_fp1 = NULL;
			gemmsup_ker_ft  ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m0 == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_sgemmsup_rv_haswell_asm_4x16n;
				ker_fp2 = bli_sgemmsup_rv_haswell_asm_3x16n;
			}
			else if ( m0 == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_sgemmsup_rv_haswell_asm_4x16n;
				ker_fp2 = bli_sgemmsup_rv_haswell_asm_4x16n;
			}
			else // if ( m0 == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_sgemmsup_rv_haswell_asm_4x16n;
				ker_fp2 = bli_sgemmsup_rv_haswell_asm_5x16n;
			}

			ker_fp1
			(
			  conja, conjb, mr1, n0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, n0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		gemmsup_ker_ft  ker_fps[6] =
		{
		  NULL,
		  bli_sgemmsup_rv_haswell_asm_1x16n,
		  bli_sgemmsup_rv_haswell_asm_2x16n,
		  bli_sgemmsup_rv_haswell_asm_3x16n,
		  bli_sgemmsup_rv_haswell_asm_4x16n,
		  bli_sgemmsup_rv_haswell_asm_5x16n
		};

		gemmsup_ker_ft  ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, n0, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t n_iter = n0 / 16;
	uint64_t n_left = n0 % 16;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b4  = ps_b * sizeof( float );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.SLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorps ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorps(ymm4,  ymm4,  ymm4)
	vxorps(ymm5,  ymm5,  ymm5)
	vxorps(ymm6,  ymm6,  ymm6)
	vxorps(ymm7,  ymm7,  ymm7)
	vxorps(ymm8,  ymm8,  ymm8)
	vxorps(ymm9,  ymm9,  ymm9)
	vxorps(ymm10, ymm10, ymm10)
	vxorps(ymm11, ymm11, ymm11)
	vxorps(ymm12, ymm12, ymm12)
	vxorps(ymm13, ymm13, ymm13)
	vxorps(ymm14, ymm14, ymm14)
	vxorps(ymm15, ymm15, ymm15)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,        15*4)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1,15*4)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2,15*4)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,        15*4)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1,15*4)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2,15*4)) // prefetch c + 5*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
	prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
	prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
	prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
	lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rcx, 1, 5*4)) // prefetch c + 7*cs_c
	prefetch(0, mem(rdx, rsi, 4, 5*4)) // prefetch c + 8*cs_c
	lea(mem(r12, rsi, 8), rdx)         // rdx = c + 8*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 9*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 10*cs_c
	prefetch(0, mem(rdx, rcx, 1, 5*4)) // prefetch c + 11*cs_c
	prefetch(0, mem(rdx, rsi, 4, 5*4)) // prefetch c + 12*cs_c
	lea(mem(r12, rcx, 4), rdx)         // rdx = c + 12*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 13*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 14*cs_c
	prefetch(0, mem(rdx, rcx, 1, 5*4)) // prefetch c + 15*cs_c

	label(.SPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b4), rdx)               // load ps_b4
	lea(mem(rbx, rdx, 1), rdx)         // rdx = a + ps_b4
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of b.
#else
	lea(mem(rbx, r8,  8), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  8), rdx)         // from next upanel of b.
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	vbroadcastss(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	vbroadcastss(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	vbroadcastss(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	vbroadcastss(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	vbroadcastss(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

	vmulps(ymm0, ymm4, ymm4)           // scale by alpha
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm0, ymm6, ymm6)
	vmulps(ymm0, ymm7, ymm7)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm0, ymm9, ymm9)
	vmulps(ymm0, ymm10, ymm10)
	vmulps(ymm0, ymm11, ymm11)
	vmulps(ymm0, ymm12, ymm12)
	vmulps(ymm0, ymm13, ymm13)
	vmulps(ymm0, ymm14, ymm14)
	vmulps(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm4)
	vmovups(ymm4, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm5)
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm6)
	vmovups(ymm6, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm7)
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm8)
	vmovups(ymm8, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm9)
	vmovups(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm10)
	vmovups(ymm10, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm11)
	vmovups(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm12)
	vmovups(ymm12, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm13)
	vmovups(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm14)
	vmovups(ymm14, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm15)
	vmovups(ymm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vunpcklps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbx, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )


	vunpckhps(ymm6, ymm4, ymm0)
	vunpckhps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rax, 2), xmm3, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbp, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c

	vunpcklps(ymm14, ymm12, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rdx        ), xmm1, xmm1)
	vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx        ))    // store ( gamma40..gamma50 )
	vmovhpd(xmm0, mem(rdx, rsi, 1))    // store ( gamma41..gamma51 )
	vmovlpd(mem(rdx, rsi, 4), xmm1, xmm1)
	vmovhpd(mem(rdx, rbx, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rdx, rsi, 4))    // store ( gamma44..gamma54 )
	vmovhpd(xmm2, mem(rdx, rbx, 1))    // store ( gamma45..gamma55 )

	vunpckhps(ymm14, ymm12, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx, rsi, 2))    // store ( gamma42..gamma52 )
	vmovhpd(xmm0, mem(rdx, rax, 1))    // store ( gamma43..gamma53 )
	vmovlpd(mem(rdx, rax, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rbp, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rdx, rax, 2))    // store ( gamma46..gamma56 )
	vmovhpd(xmm2, mem(rdx, rbp, 1))    // store ( gamma47..gamma57 )

	lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vunpcklps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbx, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )


	vunpckhps(ymm7, ymm5, ymm0)
	vunpckhps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rax, 2), xmm3, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbp, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c

	vunpcklps(ymm15, ymm13, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rdx        ), xmm1, xmm1)
	vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx        ))    // store ( gamma40..gamma50 )
	vmovhpd(xmm0, mem(rdx, rsi, 1))    // store ( gamma41..gamma51 )
	vmovlpd(mem(rdx, rsi, 4), xmm1, xmm1)
	vmovhpd(mem(rdx, rbx, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rdx, rsi, 4))    // store ( gamma44..gamma54 )
	vmovhpd(xmm2, mem(rdx, rbx, 1))    // store ( gamma45..gamma55 )

	vunpckhps(ymm15, ymm13, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx, rsi, 2))    // store ( gamma42..gamma52 )
	vmovhpd(xmm0, mem(rdx, rax, 1))    // store ( gamma43..gamma53 )
	vmovlpd(mem(rdx, rax, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rbp, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rdx, rax, 2))    // store ( gamma46..gamma56 )
	vmovhpd(xmm2, mem(rdx, rbp, 1))    // store ( gamma47..gamma57 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(ymm4, mem(rcx, 0*32))
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm6, mem(rcx, 0*32))
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm8, mem(rcx, 0*32))
	vmovups(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm10, mem(rcx, 0*32))
	vmovups(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm12, mem(rcx, 0*32))
	vmovups(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm14, mem(rcx, 0*32))
	vmovups(ymm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vunpcklps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )


	vunpckhps(ymm6, ymm4, ymm0)
	vunpckhps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c

	vunpcklps(ymm14, ymm12, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rdx        ))    // store ( gamma40..gamma50 )
	vmovhpd(xmm0, mem(rdx, rsi, 1))    // store ( gamma41..gamma51 )
	vmovlpd(xmm2, mem(rdx, rsi, 4))    // store ( gamma44..gamma54 )
	vmovhpd(xmm2, mem(rdx, rbx, 1))    // store ( gamma45..gamma55 )

	vunpckhps(ymm14, ymm12, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rdx, rsi, 2))    // store ( gamma42..gamma52 )
	vmovhpd(xmm0, mem(rdx, rax, 1))    // store ( gamma43..gamma53 )
	vmovlpd(xmm2, mem(rdx, rax, 2))    // store ( gamma46..gamma56 )
	vmovhpd(xmm2, mem(rdx, rbp, 1))    // store ( gamma47..gamma57 )

	lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vunpcklps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )


	vunpckhps(ymm7, ymm5, ymm0)
	vunpckhps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c

	vunpcklps(ymm15, ymm13, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rdx        ))    // store ( gamma40..gamma50 )
	vmovhpd(xmm0, mem(rdx, rsi, 1))    // store ( gamma41..gamma51 )
	vmovlpd(xmm2, mem(rdx, rsi, 4))    // store ( gamma44..gamma54 )
	vmovhpd(xmm2, mem(rdx, rbx, 1))    // store ( gamma45..gamma55 )

	vunpckhps(ymm15, ymm13, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rdx, rsi, 2))    // store ( gamma42..gamma52 )
	vmovhpd(xmm0, mem(rdx, rax, 1))    // store ( gamma43..gamma53 )
	vmovlpd(xmm2, mem(rdx, rax, 2))    // store ( gamma46..gamma56 )
	vmovhpd(xmm2, mem(rdx, rbp, 1))    // store ( gamma47..gamma57 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c




	label(.SDONE)




	lea(mem(r12, rsi, 8), r12)         //
	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 16*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b4), rbx)               // load ps_b4
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b4

	dec(r11)                           // jj -= 1;
	jne(.SLOOP6X8J)                    // iterate again if jj != 0.




	label(.SRETURN)



    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [ps_b4]  "m" (ps_b4),
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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 6;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		      float* cij = ( float* )c + j_edge*cs_c;
		const float* ai  = ( float* )a;
		//const float* bj  = ( float* )b + j_edge*cs_b;
		//const float* bj  = ( float* )b + ( j_edge / 8 ) * ps_b;
		const float* bj  = ( float* )b + n_iter * ps_b;

		if ( 12 <= n_left )
		{
			const dim_t nr_cur = 12;

			bli_sgemmsup_rv_haswell_asm_6x12
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 8 <= n_left )
		{
			const dim_t nr_cur = 8;

			bli_sgemmsup_rv_haswell_asm_6x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_sgemmsup_rv_haswell_asm_6x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_sgemmsup_rv_haswell_asm_6x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_sgemmsup_rv_haswell_asm_6x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_sgemmsup_r_haswell_ref_6x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_sgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_sgemmsup_rv_haswell_asm_5x16n
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

	uint64_t n_iter = n0 / 16;
	uint64_t n_left = n0 % 16;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b4  = ps_b * sizeof( float );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.SLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorps ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorps(ymm4,  ymm4,  ymm4)
	vxorps(ymm5,  ymm5,  ymm5)
	vxorps(ymm6,  ymm6,  ymm6)
	vxorps(ymm7,  ymm7,  ymm7)
	vxorps(ymm8,  ymm8,  ymm8)
	vxorps(ymm9,  ymm9,  ymm9)
	vxorps(ymm10, ymm10, ymm10)
	vxorps(ymm11, ymm11, ymm11)
	vxorps(ymm12, ymm12, ymm12)
	vxorps(ymm13, ymm13, ymm13)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,        15*4)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1,15*4)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2,15*4)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,        15*4)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1,15*4)) // prefetch c + 4*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
	prefetch(0, mem(r12,         4*4)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 4*4)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 4*4)) // prefetch c + 2*cs_c
	prefetch(0, mem(r12, rcx, 1, 4*4)) // prefetch c + 3*cs_c
	prefetch(0, mem(r12, rsi, 4, 4*4)) // prefetch c + 4*cs_c
	lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 4*4)) // prefetch c + 5*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*4)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rcx, 1, 4*4)) // prefetch c + 7*cs_c
	prefetch(0, mem(rdx, rsi, 4, 4*4)) // prefetch c + 8*cs_c
	lea(mem(r12, rsi, 8), rdx)         // rdx = c + 8*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 4*4)) // prefetch c + 9*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*4)) // prefetch c + 10*cs_c
	prefetch(0, mem(rdx, rcx, 1, 4*4)) // prefetch c + 11*cs_c
	prefetch(0, mem(rdx, rsi, 4, 4*4)) // prefetch c + 12*cs_c
	lea(mem(r12, rcx, 4), rdx)         // rdx = c + 12*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 4*4)) // prefetch c + 13*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*4)) // prefetch c + 14*cs_c
	prefetch(0, mem(rdx, rcx, 1, 4*4)) // prefetch c + 15*cs_c

	label(.SPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b4), rdx)               // load ps_b4
	lea(mem(rbx, rdx, 1), rdx)         // rdx = a + ps_b4
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of b.
#else
	lea(mem(rbx, r8,  8), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  8), rdx)         // from next upanel of b.
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

	vmulps(ymm0, ymm4, ymm4)           // scale by alpha
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm0, ymm6, ymm6)
	vmulps(ymm0, ymm7, ymm7)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm0, ymm9, ymm9)
	vmulps(ymm0, ymm10, ymm10)
	vmulps(ymm0, ymm11, ymm11)
	vmulps(ymm0, ymm12, ymm12)
	vmulps(ymm0, ymm13, ymm13)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm4)
	vmovups(ymm4, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm5)
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm6)
	vmovups(ymm6, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm7)
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm8)
	vmovups(ymm8, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm9)
	vmovups(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm10)
	vmovups(ymm10, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm11)
	vmovups(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm12)
	vmovups(ymm12, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm13)
	vmovups(ymm13, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vunpcklps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbx, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm6, ymm4, ymm0)
	vunpckhps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rax, 2), xmm3, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbp, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm12, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

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

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(mem(rdx, rsi, 4), xmm1)
	vmovss(mem(rdx, rbx, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(mem(rdx, rax, 2), xmm1)
	vmovss(mem(rdx, rbp, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vunpcklps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbx, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm7, ymm5, ymm0)
	vunpckhps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rax, 2), xmm3, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbp, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm13, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

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

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(mem(rdx, rsi, 4), xmm1)
	vmovss(mem(rdx, rbx, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(mem(rdx, rax, 2), xmm1)
	vmovss(mem(rdx, rbp, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(ymm4, mem(rcx, 0*32))
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm6, mem(rcx, 0*32))
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm8, mem(rcx, 0*32))
	vmovups(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm10, mem(rcx, 0*32))
	vmovups(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm12, mem(rcx, 0*32))
	vmovups(ymm13, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vunpcklps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm6, ymm4, ymm0)
	vunpckhps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm12, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vunpcklps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm7, ymm5, ymm0)
	vunpckhps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm13, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c




	label(.SDONE)




	lea(mem(r12, rsi, 8), r12)         //
	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 16*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b4), rbx)               // load ps_b4
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b4

	dec(r11)                           // jj -= 1;
	jne(.SLOOP6X8J)                    // iterate again if jj != 0.




	label(.SRETURN)



    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [ps_b4]  "m" (ps_b4),
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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 5;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		      float* cij = ( float* )c + j_edge*cs_c;
		const float* ai  = ( float* )a;
		//const float* bj  = ( float* )b + j_edge*cs_b;
		//const float* bj  = ( float* )b + ( j_edge / 8 ) * ps_b;
		const float* bj  = ( float* )b + n_iter * ps_b;

		if ( 12 <= n_left )
		{
			const dim_t nr_cur = 12;

			bli_sgemmsup_rv_haswell_asm_5x12
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 8 <= n_left )
		{
			const dim_t nr_cur = 8;

			bli_sgemmsup_rv_haswell_asm_5x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_sgemmsup_rv_haswell_asm_5x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_sgemmsup_rv_haswell_asm_5x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_sgemmsup_rv_haswell_asm_5x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_sgemmsup_r_haswell_ref_5x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_sgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_sgemmsup_rv_haswell_asm_4x16n
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

	uint64_t n_iter = n0 / 16;
	uint64_t n_left = n0 % 16;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b4  = ps_b * sizeof( float );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.SLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorps ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorps(ymm4,  ymm4,  ymm4)
	vxorps(ymm5,  ymm5,  ymm5)
	vxorps(ymm6,  ymm6,  ymm6)
	vxorps(ymm7,  ymm7,  ymm7)
	vxorps(ymm8,  ymm8,  ymm8)
	vxorps(ymm9,  ymm9,  ymm9)
	vxorps(ymm10, ymm10, ymm10)
	vxorps(ymm11, ymm11, ymm11)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,        15*4)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1,15*4)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2,15*4)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,        15*4)) // prefetch c + 3*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
	prefetch(0, mem(r12,         3*4)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 3*4)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 3*4)) // prefetch c + 2*cs_c
	prefetch(0, mem(r12, rcx, 1, 3*4)) // prefetch c + 3*cs_c
	prefetch(0, mem(r12, rsi, 4, 3*4)) // prefetch c + 4*cs_c
	lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 3*4)) // prefetch c + 5*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*4)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rcx, 1, 3*4)) // prefetch c + 7*cs_c
	prefetch(0, mem(rdx, rsi, 4, 3*4)) // prefetch c + 8*cs_c
	lea(mem(r12, rsi, 8), rdx)         // rdx = c + 8*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 3*4)) // prefetch c + 9*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*4)) // prefetch c + 10*cs_c
	prefetch(0, mem(rdx, rcx, 1, 3*4)) // prefetch c + 11*cs_c
	prefetch(0, mem(rdx, rsi, 4, 3*4)) // prefetch c + 12*cs_c
	lea(mem(r12, rcx, 4), rdx)         // rdx = c + 12*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 3*4)) // prefetch c + 13*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*4)) // prefetch c + 14*cs_c
	prefetch(0, mem(rdx, rcx, 1, 3*4)) // prefetch c + 15*cs_c

	label(.SPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b4), rdx)               // load ps_b4
	lea(mem(rbx, rdx, 1), rdx)         // rdx = a + ps_b4
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of b.
#else
	lea(mem(rbx, r8,  8), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  8), rdx)         // from next upanel of b.
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	vbroadcastss(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

	vmulps(ymm0, ymm4, ymm4)           // scale by alpha
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm0, ymm6, ymm6)
	vmulps(ymm0, ymm7, ymm7)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm0, ymm9, ymm9)
	vmulps(ymm0, ymm10, ymm10)
	vmulps(ymm0, ymm11, ymm11)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm4)
	vmovups(ymm4, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm5)
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm6)
	vmovups(ymm6, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm7)
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm8)
	vmovups(ymm8, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm9)
	vmovups(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm10)
	vmovups(ymm10, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm11)
	vmovups(ymm11, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vunpcklps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbx, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm6, ymm4, ymm0)
	vunpckhps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rax, 2), xmm3, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbp, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vunpcklps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx        ), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbx, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm7, ymm5, ymm0)
	vunpckhps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
	vfmadd231ps(mem(rcx, rax, 2), xmm3, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vfmadd231ps(mem(rcx, rax, 1), xmm3, xmm1)
	vfmadd231ps(mem(rcx, rbp, 1), xmm3, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(ymm4, mem(rcx, 0*32))
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm6, mem(rcx, 0*32))
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm8, mem(rcx, 0*32))
	vmovups(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm10, mem(rcx, 0*32))
	vmovups(ymm11, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vunpcklps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm6, ymm4, ymm0)
	vunpckhps(ymm10, ymm8, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vunpcklps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx        ))    // store ( gamma00..gamma30 )
	vmovups(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma34 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rsi, 1))    // store ( gamma01..gamma31 )
	vmovups(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma35 )

	vunpckhps(ymm7, ymm5, ymm0)
	vunpckhps(ymm11, ymm9, ymm1)
	vshufps(imm(0x4e), ymm1, ymm0, ymm2)
	vblendps(imm(0xcc), ymm2, ymm0, ymm0)
	vblendps(imm(0x33), ymm2, ymm1, ymm1)

	vextractf128(imm(0x1), ymm0, xmm2)
	vmovups(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma32 )
	vmovups(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma36 )

	vextractf128(imm(0x1), ymm1, xmm2)
	vmovups(xmm1, mem(rcx, rax, 1))    // store ( gamma03..gamma33 )
	vmovups(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma37 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c




	label(.SDONE)




	lea(mem(r12, rsi, 8), r12)         //
	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 16*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b4), rbx)               // load ps_b4
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b4

	dec(r11)                           // jj -= 1;
	jne(.SLOOP6X8J)                    // iterate again if jj != 0.




	label(.SRETURN)



    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [ps_b4]  "m" (ps_b4),
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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 4;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		      float* cij = ( float* )c + j_edge*cs_c;
		const float* ai  = ( float* )a;
		//const float* bj  = ( float* )b + j_edge*cs_b;
		//const float* bj  = ( float* )b + ( j_edge / 8 ) * ps_b;
		const float* bj  = ( float* )b + n_iter * ps_b;

		if ( 12 <= n_left )
		{
			const dim_t nr_cur = 12;

			bli_sgemmsup_rv_haswell_asm_4x12
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 8 <= n_left )
		{
			const dim_t nr_cur = 8;

			bli_sgemmsup_rv_haswell_asm_4x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_sgemmsup_rv_haswell_asm_4x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_sgemmsup_rv_haswell_asm_4x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_sgemmsup_rv_haswell_asm_4x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_sgemmsup_r_haswell_ref_4x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_sgemmsup_rv_haswell_asm_3x16n
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

	uint64_t n_iter = n0 / 16;
	uint64_t n_left = n0 % 16;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b4  = ps_b * sizeof( float );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.SLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorps ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorps(ymm4,  ymm4,  ymm4)
	vxorps(ymm5,  ymm5,  ymm5)
	vxorps(ymm6,  ymm6,  ymm6)
	vxorps(ymm7,  ymm7,  ymm7)
	vxorps(ymm8,  ymm8,  ymm8)
	vxorps(ymm9,  ymm9,  ymm9)
	vxorps(ymm10, ymm10, ymm10)
	vxorps(ymm11, ymm11, ymm11)
	vxorps(ymm12, ymm12, ymm12)
	vxorps(ymm13, ymm13, ymm13)
	vxorps(ymm14, ymm14, ymm14)
	vxorps(ymm15, ymm15, ymm15)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(r12, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,        15*4)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1,15*4)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2,15*4)) // prefetch c + 2*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
	prefetch(0, mem(r12,         2*4)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 2*4)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 2*4)) // prefetch c + 2*cs_c
	prefetch(0, mem(r12, rcx, 1, 2*4)) // prefetch c + 3*cs_c
	prefetch(0, mem(r12, rsi, 4, 2*4)) // prefetch c + 4*cs_c
	lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 2*4)) // prefetch c + 5*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*4)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rcx, 1, 2*4)) // prefetch c + 7*cs_c
	prefetch(0, mem(rdx, rsi, 4, 2*4)) // prefetch c + 8*cs_c
	lea(mem(r12, rsi, 8), rdx)         // rdx = c + 8*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 2*4)) // prefetch c + 9*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*4)) // prefetch c + 10*cs_c
	prefetch(0, mem(rdx, rcx, 1, 2*4)) // prefetch c + 11*cs_c
	prefetch(0, mem(rdx, rsi, 4, 2*4)) // prefetch c + 12*cs_c
	lea(mem(r12, rcx, 4), rdx)         // rdx = c + 12*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 2*4)) // prefetch c + 13*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*4)) // prefetch c + 14*cs_c
	prefetch(0, mem(rdx, rcx, 1, 2*4)) // prefetch c + 15*cs_c

	label(.SPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b4), rdx)               // load ps_b4
	lea(mem(rbx, rdx, 1), rdx)         // rdx = a + ps_b4
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of b.
#else
	lea(mem(rbx, r8,  8), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  8), rdx)         // from next upanel of b.
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

	vmulps(ymm0, ymm4, ymm4)           // scale by alpha
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm0, ymm6, ymm6)
	vmulps(ymm0, ymm7, ymm7)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm0, ymm9, ymm9)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm4)
	vmovups(ymm4, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm5)
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm6)
	vmovups(ymm6, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm7)
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm8)
	vmovups(ymm8, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm9)
	vmovups(ymm9, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx        ), xmm1, xmm1)
	vmovhpd(mem(rcx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(mem(rcx, rsi, 4), xmm1, xmm1)
	vmovhpd(mem(rcx, rbx, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(mem(rcx, rax, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rbp, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm8, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

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

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(mem(rdx, rsi, 4), xmm1)
	vmovss(mem(rdx, rbx, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(mem(rdx, rax, 2), xmm1)
	vmovss(mem(rdx, rbp, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx        ), xmm1, xmm1)
	vmovhpd(mem(rcx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(mem(rcx, rsi, 4), xmm1, xmm1)
	vmovhpd(mem(rcx, rbx, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(mem(rcx, rax, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rbp, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm9, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

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

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(mem(rdx, rsi, 4), xmm1)
	vmovss(mem(rdx, rbx, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(mem(rdx, rax, 2), xmm1)
	vmovss(mem(rdx, rbp, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	//lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(ymm4, mem(rcx, 0*32))
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm6, mem(rcx, 0*32))
	vmovups(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovups(ymm8, mem(rcx, 0*32))
	vmovups(ymm9, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm8, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	lea(mem(rdx, rsi, 8), rdx)         // rdx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	vmovups(ymm9, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rdx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rdx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rdx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rdx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(xmm2, mem(rdx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rdx, rbp, 1))     // store ( gamma47 )

	//lea(mem(rcx, rsi, 4), rcx)




	label(.SDONE)




	lea(mem(r12, rsi, 8), r12)         //
	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 16*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b4), rbx)               // load ps_b4
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b4

	dec(r11)                           // jj -= 1;
	jne(.SLOOP6X8J)                    // iterate again if jj != 0.




	label(.SRETURN)



    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [ps_b4]  "m" (ps_b4),
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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		      float* cij = ( float* )c + j_edge*cs_c;
		const float* ai  = ( float* )a;
		//const float* bj  = ( float* )b + j_edge*cs_b;
		//const float* bj  = ( float* )b + ( j_edge / 8 ) * ps_b;
		const float* bj  = ( float* )b + n_iter * ps_b;

		if ( 12 <= n_left )
		{
			const dim_t nr_cur = 12;

			bli_sgemmsup_rv_haswell_asm_3x12
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 8 <= n_left )
		{
			const dim_t nr_cur = 8;

			bli_sgemmsup_rv_haswell_asm_3x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_sgemmsup_rv_haswell_asm_3x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_sgemmsup_rv_haswell_asm_3x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_sgemmsup_rv_haswell_asm_3x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_sgemmsup_r_haswell_ref_3x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_sgemmsup_rv_haswell_asm_2x16n
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

	uint64_t n_iter = n0 / 16;
	uint64_t n_left = n0 % 16;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b4  = ps_b * sizeof( float );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.SLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorps ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorps(ymm4,  ymm4,  ymm4)
	vxorps(ymm5,  ymm5,  ymm5)
	vxorps(ymm6,  ymm6,  ymm6)
	vxorps(ymm7,  ymm7,  ymm7)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(r12, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,        15*4)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1,15*4)) // prefetch c + 1*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
	prefetch(0, mem(r12,         1*4)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 1*4)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 1*4)) // prefetch c + 2*cs_c
	prefetch(0, mem(r12, rcx, 1, 1*4)) // prefetch c + 3*cs_c
	prefetch(0, mem(r12, rsi, 4, 1*4)) // prefetch c + 4*cs_c
	lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 1*4)) // prefetch c + 5*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*4)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rcx, 1, 1*4)) // prefetch c + 7*cs_c
	prefetch(0, mem(rdx, rsi, 4, 1*4)) // prefetch c + 8*cs_c
	lea(mem(r12, rsi, 8), rdx)         // rdx = c + 8*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 1*4)) // prefetch c + 9*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*4)) // prefetch c + 10*cs_c
	prefetch(0, mem(rdx, rcx, 1, 1*4)) // prefetch c + 11*cs_c
	prefetch(0, mem(rdx, rsi, 4, 1*4)) // prefetch c + 12*cs_c
	lea(mem(r12, rcx, 4), rdx)         // rdx = c + 12*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 1*4)) // prefetch c + 13*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*4)) // prefetch c + 14*cs_c
	prefetch(0, mem(rdx, rcx, 1, 1*4)) // prefetch c + 15*cs_c

	label(.SPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b4), rdx)               // load ps_b4
	lea(mem(rbx, rdx, 1), rdx)         // rdx = a + ps_b4
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of b.
#else
	lea(mem(rbx, r8,  8), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  8), rdx)         // from next upanel of b.
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	vbroadcastss(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

	vmulps(ymm0, ymm4, ymm4)           // scale by alpha
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm0, ymm6, ymm6)
	vmulps(ymm0, ymm7, ymm7)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm4)
	vmovups(ymm4, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm5)
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm6)
	vmovups(ymm6, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm7)
	vmovups(ymm7, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx        ), xmm1, xmm1)
	vmovhpd(mem(rcx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(mem(rcx, rsi, 4), xmm1, xmm1)
	vmovhpd(mem(rcx, rbx, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(mem(rcx, rax, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rbp, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx        ), xmm1, xmm1)
	vmovhpd(mem(rcx, rsi, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(mem(rcx, rsi, 4), xmm1, xmm1)
	vmovhpd(mem(rcx, rbx, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm0)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(mem(rcx, rax, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rbp, 1), xmm1, xmm1)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(ymm4, mem(rcx, 0*32))
	vmovups(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovups(ymm6, mem(rcx, 0*32))
	vmovups(ymm7, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)

	                                   // begin I/O on columns 0-7
	vunpcklps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm6, ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vunpcklps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx        ))    // store ( gamma00..gamma10 )
	vmovhpd(xmm0, mem(rcx, rsi, 1))    // store ( gamma01..gamma11 )
	vmovlpd(xmm2, mem(rcx, rsi, 4))    // store ( gamma04..gamma14 )
	vmovhpd(xmm2, mem(rcx, rbx, 1))    // store ( gamma05..gamma15 )

	vunpckhps(ymm7, ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm2)
	vmovlpd(xmm0, mem(rcx, rsi, 2))    // store ( gamma02..gamma12 )
	vmovhpd(xmm0, mem(rcx, rax, 1))    // store ( gamma03..gamma13 )
	vmovlpd(xmm2, mem(rcx, rax, 2))    // store ( gamma06..gamma16 )
	vmovhpd(xmm2, mem(rcx, rbp, 1))    // store ( gamma07..gamma17 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c




	label(.SDONE)




	lea(mem(r12, rsi, 8), r12)         //
	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 16*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b4), rbx)               // load ps_b4
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b4

	dec(r11)                           // jj -= 1;
	jne(.SLOOP6X8J)                    // iterate again if jj != 0.




	label(.SRETURN)



    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [ps_b4]  "m" (ps_b4),
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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 2;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		      float* cij = ( float* )c + j_edge*cs_c;
		const float* ai  = ( float* )a;
		//const float* bj  = ( float* )b + j_edge*cs_b;
		//const float* bj  = ( float* )b + ( j_edge / 8 ) * ps_b;
		const float* bj  = ( float* )b + n_iter * ps_b;

		if ( 12 <= n_left )
		{
			const dim_t nr_cur = 12;

			bli_sgemmsup_rv_haswell_asm_2x12
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 8 <= n_left )
		{
			const dim_t nr_cur = 8;

			bli_sgemmsup_rv_haswell_asm_2x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_sgemmsup_rv_haswell_asm_2x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_sgemmsup_rv_haswell_asm_2x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_sgemmsup_rv_haswell_asm_2x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_sgemmsup_r_haswell_ref_2x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_sgemmsup_rv_haswell_asm_1x16n
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

	uint64_t n_iter = n0 / 16;
	uint64_t n_left = n0 % 16;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b4  = ps_b * sizeof( float );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
	lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)
	//lea(mem(, r11, 4), r11)            // cs_b *= sizeof(float)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.SLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorps ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorps(ymm4,  ymm4,  ymm4)
	vxorps(ymm5,  ymm5,  ymm5)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLPFETCH)                    // jump to column storage case
	label(.SROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(r12, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,        15*4)) // prefetch c + 0*rs_c

	jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
	label(.SCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
	lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
	prefetch(0, mem(r12,         0*4)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 0*4)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 0*4)) // prefetch c + 2*cs_c
	prefetch(0, mem(r12, rcx, 1, 0*4)) // prefetch c + 3*cs_c
	prefetch(0, mem(r12, rsi, 4, 0*4)) // prefetch c + 4*cs_c
	lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 0*4)) // prefetch c + 5*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*4)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rcx, 1, 0*4)) // prefetch c + 7*cs_c
	prefetch(0, mem(rdx, rsi, 4, 0*4)) // prefetch c + 8*cs_c
	lea(mem(r12, rsi, 8), rdx)         // rdx = c + 8*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 0*4)) // prefetch c + 9*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*4)) // prefetch c + 10*cs_c
	prefetch(0, mem(rdx, rcx, 1, 0*4)) // prefetch c + 11*cs_c
	prefetch(0, mem(rdx, rsi, 4, 0*4)) // prefetch c + 12*cs_c
	lea(mem(r12, rcx, 4), rdx)         // rdx = c + 12*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 0*4)) // prefetch c + 13*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*4)) // prefetch c + 14*cs_c
	prefetch(0, mem(rdx, rcx, 1, 0*4)) // prefetch c + 15*cs_c

	label(.SPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b4), rdx)               // load ps_b4
	lea(mem(rbx, rdx, 1), rdx)         // rdx = a + ps_b4
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of b.
#else
	lea(mem(rbx, r8,  8), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  8), rdx)         // from next upanel of b.
	lea(mem(r10, r10, 2), rcx)         // rcx = 3*rs_b;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.SLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)



	dec(rsi)                           // i -= 1;
	jne(.SLOOPKITER)                   // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)
#endif

	vmovups(mem(rbx, 0*32), ymm0)
	vmovups(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastss(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)


	dec(rsi)                           // i -= 1;
	jne(.SLOOPKLEFT)                   // iterate again if i != 0.



	label(.SPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

	vmulps(ymm0, ymm4, ymm4)           // scale by alpha
	vmulps(ymm0, ymm5, ymm5)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;
	lea(mem(rax, rsi, 4), rbp)         // rbp = 7*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
	je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORED)                    // jump to column storage case



	label(.SROWSTORED)


	vfmadd231ps(mem(rcx, 0*32), ymm3, ymm4)
	vmovups(ymm4, mem(rcx, 0*32))

	vfmadd231ps(mem(rcx, 1*32), ymm3, ymm5)
	vmovups(ymm5, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORED)

	                                   // begin I/O on columns 0-7
	vmovups(ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(mem(rcx        ), xmm1)
	vmovss(mem(rcx, rsi, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rcx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(mem(rcx, rsi, 2), xmm1)
	vmovss(mem(rcx, rax, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rcx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(mem(rcx, rsi, 4), xmm1)
	vmovss(mem(rcx, rbx, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rcx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(mem(rcx, rax, 2), xmm1)
	vmovss(mem(rcx, rbp, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rcx, rbp, 1))     // store ( gamma47 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vmovups(ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(mem(rcx        ), xmm1)
	vmovss(mem(rcx, rsi, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rcx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(mem(rcx, rsi, 2), xmm1)
	vmovss(mem(rcx, rax, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rcx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(mem(rcx, rsi, 4), xmm1)
	vmovss(mem(rcx, rbx, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rcx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(mem(rcx, rax, 2), xmm1)
	vmovss(mem(rcx, rbp, 1), xmm6)
	vfmadd231ps(xmm1, xmm3, xmm2)
	vfmadd231ps(xmm6, xmm3, xmm4)
	vmovss(xmm2, mem(rcx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rcx, rbp, 1))     // store ( gamma47 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	jmp(.SDONE)                        // jump to end.




	label(.SBETAZERO)


	cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
	jz(.SCOLSTORBZ)                    // jump to column storage case



	label(.SROWSTORBZ)


	vmovups(ymm4, mem(rcx, 0*32))
	vmovups(ymm5, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.SDONE)                        // jump to end.



	label(.SCOLSTORBZ)


	                                   // begin I/O on columns 0-7
	vmovups(ymm4, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rcx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rcx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rcx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rcx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(xmm2, mem(rcx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rcx, rbp, 1))     // store ( gamma47 )

	lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c


	                                   // begin I/O on columns 8-15
	vmovups(ymm5, ymm0)
	vextractf128(imm(0x1), ymm0, xmm8)

	vpermilps(imm(0xe4), xmm0, xmm2)
	vpermilps(imm(0x39), xmm0, xmm4)
	vmovss(xmm2, mem(rcx        ))     // store ( gamma40 )
	vmovss(xmm4, mem(rcx, rsi, 1))     // store ( gamma41 )

	vpermilps(imm(0x4e), xmm0, xmm2)
	vpermilps(imm(0x93), xmm0, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 2))     // store ( gamma42 )
	vmovss(xmm4, mem(rcx, rax, 1))     // store ( gamma43 )

	vpermilps(imm(0xe4), xmm8, xmm2)
	vpermilps(imm(0x39), xmm8, xmm4)
	vmovss(xmm2, mem(rcx, rsi, 4))     // store ( gamma44 )
	vmovss(xmm4, mem(rcx, rbx, 1))     // store ( gamma45 )

	vpermilps(imm(0x4e), xmm8, xmm2)
	vpermilps(imm(0x93), xmm8, xmm4)
	vmovss(xmm2, mem(rcx, rax, 2))     // store ( gamma46 )
	vmovss(xmm4, mem(rcx, rbp, 1))     // store ( gamma47 )

	//lea(mem(rcx, rsi, 8), rcx)         // rcx += 8*cs_c




	label(.SDONE)




	lea(mem(r12, rsi, 8), r12)         //
	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 16*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b4), rbx)               // load ps_b4
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b4

	dec(r11)                           // jj -= 1;
	jne(.SLOOP6X8J)                    // iterate again if jj != 0.




	label(.SRETURN)



    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [ps_b4]  "m" (ps_b4),
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

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 1;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		      float* cij = ( float* )c + j_edge*cs_c;
		const float* ai  = ( float* )a;
		//const float* bj  = ( float* )b + j_edge*cs_b;
		//const float* bj  = ( float* )b + ( j_edge / 8 ) * ps_b;
		const float* bj  = ( float* )b + n_iter * ps_b;

		if ( 12 <= n_left )
		{
			const dim_t nr_cur = 12;

			bli_sgemmsup_rv_haswell_asm_1x12
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 8 <= n_left )
		{
			const dim_t nr_cur = 8;

			bli_sgemmsup_rv_haswell_asm_1x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_sgemmsup_rv_haswell_asm_1x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_sgemmsup_rv_haswell_asm_1x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_sgemmsup_rv_haswell_asm_1x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_sgemmsup_r_haswell_ref_1x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

