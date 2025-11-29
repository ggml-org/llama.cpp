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
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )


void bli_dgemmsup_rv_haswell_asm_6x8n
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t m_left = m0 % 6;

	// First check whether this is a edge case in the m dimension. If so,
	// dispatch other ?x8m kernels, as needed.
	if ( m_left )
	{
		      double* cij = ( double* )c;
		const double* bj  = ( double* )b;
		const double* ai  = ( double* )a;

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
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x8n;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x8n;
			}
			else if ( m0 == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x8n;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x8n;
			}
			else // if ( m0 == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x8n;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x8n;
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
		  bli_dgemmsup_rv_haswell_asm_1x8n,
		  bli_dgemmsup_rv_haswell_asm_2x8n,
		  bli_dgemmsup_rv_haswell_asm_3x8n,
		  bli_dgemmsup_rv_haswell_asm_4x8n,
		  bli_dgemmsup_rv_haswell_asm_5x8n
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

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b8  = ps_b * sizeof( double );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.DLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b8), rdx)               // load ps_b8
	lea(mem(rbx, rdx, 1), rdx)         // rdx = b + ps_b8
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
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)                      // b_prefetch += rs_b;
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(ymm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 8*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b8), rbx)               // load ps_b8
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b8

	dec(r11)                           // jj -= 1;
	jne(.DLOOP6X8J)                    // iterate again if jj != 0.




	label(.DRETURN)



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
      [ps_b8]  "m" (ps_b8),
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

		      double* cij = ( double* )c + j_edge*cs_c;
		const double* ai  = ( double* )a;
		//const double* bj  = ( double* )b + j_edge*cs_b;
		//const double* bj  = ( double* )b + ( j_edge / 8 ) * ps_b;
		const double* bj  = ( double* )b + n_iter * ps_b;

		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_dgemmsup_rv_haswell_asm_6x6
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

			bli_dgemmsup_rv_haswell_asm_6x4
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

			bli_dgemmsup_rv_haswell_asm_6x2
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

			bli_dgemmsup_r_haswell_ref_6x1
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

void bli_dgemmsup_rv_haswell_asm_5x8n
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b8  = ps_b * sizeof( double );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.DLOOP6X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         4*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         4*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b8), rdx)               // load ps_b8
	lea(mem(rbx, rdx, 1), rdx)         // rdx = b + ps_b8
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
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)                      // b_prefetch += rs_b;
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovlpd(mem(rdx        ), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm12, ymm3, ymm0)
	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovlpd(mem(rdx        ), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm13, ymm3, ymm0)
	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(ymm13, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovupd(ymm12, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovupd(ymm13, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx        ))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 8*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b8), rbx)               // load ps_b8
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b8

	dec(r11)                           // jj -= 1;
	jne(.DLOOP6X8J)                    // iterate again if jj != 0.




	label(.DRETURN)



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
      [ps_b8]  "m" (ps_b8),
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

		      double* cij = ( double* )c + j_edge*cs_c;
		const double* ai  = ( double* )a;
		//const double* bj  = ( double* )b + j_edge*cs_b;
		//const double* bj  = ( double* )b + ( j_edge / 8 ) * ps_b;
		const double* bj  = ( double* )b + n_iter * ps_b;

		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_dgemmsup_rv_haswell_asm_5x6
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

			bli_dgemmsup_rv_haswell_asm_5x4
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

			bli_dgemmsup_rv_haswell_asm_5x2
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

			bli_dgemmsup_r_haswell_ref_5x1
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

void bli_dgemmsup_rv_haswell_asm_4x8n
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b8  = ps_b * sizeof( double );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.DLOOP4X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b8), rdx)               // load ps_b8
	lea(mem(rbx, rdx, 1), rdx)         // rdx = b + ps_b8
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
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	// ---------------------------------- iteration 3

#if 1
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)                      // b_prefetch += rs_b;
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)




	label(.DDONE)




	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 8*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b8), rbx)               // load ps_b8
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b8

	dec(r11)                           // jj -= 1;
	jne(.DLOOP4X8J)                    // iterate again if jj != 0.




	label(.DRETURN)



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
      [ps_b8]  "m" (ps_b8),
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

		      double* cij = ( double* )c + j_edge*cs_c;
		const double* ai  = ( double* )a;
		//const double* bj  = ( double* )b + j_edge*cs_b;
		//const double* bj  = ( double* )b + ( j_edge / 8 ) * ps_b;
		const double* bj  = ( double* )b + n_iter * ps_b;

		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_dgemmsup_rv_haswell_asm_4x6
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

			bli_dgemmsup_rv_haswell_asm_4x4
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

			bli_dgemmsup_rv_haswell_asm_4x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			const dim_t nr_cur = 1;

			bli_dgemmsup_r_haswell_ref_4x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
	}
}

void bli_dgemmsup_rv_haswell_asm_3x8n
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b8  = ps_b * sizeof( double );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.DLOOP4X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(r12, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         2*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 2*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 2*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         2*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b8), rdx)               // load ps_b8
	lea(mem(rbx, rdx, 1), rdx)         // rdx = b + ps_b8
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
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)                      // b_prefetch += rs_b;
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx        ), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)
	vextractf128(imm(0x1), ymm9, xmm14)
	vextractf128(imm(0x1), ymm11, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), xmm3, xmm5)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm7)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm9)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm11)
	vmovupd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	vmovupd(xmm9, mem(rcx, rsi, 2))
	vmovupd(xmm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx        ), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vmovupd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)
	vextractf128(imm(0x1), ymm9, xmm14)
	vextractf128(imm(0x1), ymm11, xmm15)

	vmovupd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	vmovupd(xmm9, mem(rcx, rsi, 2))
	vmovupd(xmm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx        ))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 8*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b8), rbx)               // load ps_b8
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b8

	dec(r11)                           // jj -= 1;
	jne(.DLOOP4X8J)                    // iterate again if jj != 0.




	label(.DRETURN)



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
      [ps_b8]  "m" (ps_b8),
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

		      double* cij = ( double* )c + j_edge*cs_c;
		const double* ai  = ( double* )a;
		//const double* bj  = ( double* )b + j_edge*cs_b;
		//const double* bj  = ( double* )b + ( j_edge / 8 ) * ps_b;
		const double* bj  = ( double* )b + n_iter * ps_b;

		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_dgemmsup_rv_haswell_asm_3x6
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

			bli_dgemmsup_rv_haswell_asm_3x4
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

			bli_dgemmsup_rv_haswell_asm_3x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			const dim_t nr_cur = 1;

			bli_dgemmsup_r_haswell_ref_3x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
	}
}

void bli_dgemmsup_rv_haswell_asm_2x8n
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b8  = ps_b * sizeof( double );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.DLOOP2X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(r12, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         1*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 1*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 1*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         1*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b8), rdx)               // load ps_b8
	lea(mem(rbx, rdx, 1), rdx)         // rdx = b + ps_b8
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
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)                      // b_prefetch += rs_b;
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rcx        ), xmm3, xmm0)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rcx        ))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rcx        ), xmm3, xmm0)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rcx        ))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rcx        ))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rcx        ))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)




	label(.DDONE)




	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 8*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b8), rbx)               // load ps_b8
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b8

	dec(r11)                           // jj -= 1;
	jne(.DLOOP2X8J)                    // iterate again if jj != 0.




	label(.DRETURN)



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
      [ps_b8]  "m" (ps_b8),
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

		      double* cij = ( double* )c + j_edge*cs_c;
		const double* ai  = ( double* )a;
		//const double* bj  = ( double* )b + j_edge*cs_b;
		//const double* bj  = ( double* )b + ( j_edge / 8 ) * ps_b;
		const double* bj  = ( double* )b + n_iter * ps_b;

		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_dgemmsup_rv_haswell_asm_2x6
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

			bli_dgemmsup_rv_haswell_asm_2x4
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

			bli_dgemmsup_rv_haswell_asm_2x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			const dim_t nr_cur = 1;

			bli_dgemmsup_r_haswell_ref_2x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
	}
}

void bli_dgemmsup_rv_haswell_asm_1x8n
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	double *a = (double *)a0;
	double *b = (double *)b0;
	double *c = (double *)c0;

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of B and convert it to units of bytes.
	uint64_t ps_b   = bli_auxinfo_ps_b( data );
	uint64_t ps_b8  = ps_b * sizeof( double );

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	//mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rbx = b
	// read rax from var(a) near beginning of loop
	// r11 = m dim index ii

	mov(var(n_iter), r11)              // jj = n_iter;

	label(.DLOOP1X8J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
#endif

	mov(var(a), rax)                   // load address of a.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rbx)                      // reset rbx to current upanel of b.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(r12, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         0*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 0*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 0*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         0*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c

#if 1
	mov(var(ps_b8), rdx)               // load ps_b8
	lea(mem(rbx, rdx, 1), rdx)         // rdx = b + ps_b8
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
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 1

#if 1
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r10, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	// ---------------------------------- iteration 3

#if 1
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r10, 4), rdx)         // b_prefetch += 4*rs_b;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r10, rdx)                      // b_prefetch += rs_b;
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vmovlpd(mem(rcx        ), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm4, ymm3, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vmovlpd(mem(rcx        ), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm5, ymm3, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vmovupd(ymm4, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vmovupd(ymm5, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx        ))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)




	label(.DDONE)




	lea(mem(r12, rsi, 8), r12)         // c_jj = r12 += 8*cs_c

	//add(imm(8*8), r14)                 // b_jj = r14 += 8*cs_b
	mov(var(ps_b8), rbx)               // load ps_b8
	lea(mem(r14, rbx, 1), r14)         // b_jj = r14 += ps_b8

	dec(r11)                           // jj -= 1;
	jne(.DLOOP1X8J)                    // iterate again if jj != 0.




	label(.DRETURN)



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
      [ps_b8]  "m" (ps_b8),
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

		      double* cij = ( double* )c + j_edge*cs_c;
		const double* ai  = ( double* )a;
		//const double* bj  = ( double* )b + j_edge*cs_b;
		//const double* bj  = ( double* )b + ( j_edge / 8 ) * ps_b;
		const double* bj  = ( double* )b + n_iter * ps_b;

		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_dgemmsup_rv_haswell_asm_1x6
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

			bli_dgemmsup_rv_haswell_asm_1x4
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

			bli_dgemmsup_rv_haswell_asm_1x2
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

			bli_dgemmsup_r_haswell_ref_1x1
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
#else
			bli_ddotxv_ex
			(
			  conja, conjb, k0,
			  alpha, ai, cs_a0, bj, rs_b0,
			  beta, cij, cntx, NULL
			);
#endif
		}
	}
}

