
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
#include "immintrin.h"

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
void bli_cgemmsup_rv_zen_asm_3x8n
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
	uint64_t m_left = m0 % 3;
	if ( m_left )
	{
		cgemmsup_ker_ft ker_fps[3] =
		{
			NULL,
			bli_cgemmsup_rv_zen_asm_1x8n,
			bli_cgemmsup_rv_zen_asm_2x8n,
		};
		cgemmsup_ker_ft ker_fp = ker_fps[ m_left ];
		ker_fp
		(
			conja, conjb, m_left, n0, k0,
			alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
			beta, c, rs_c0, cs_c0, data, cntx
		);
		return;
	}
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = k0 / 4;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------
	//scratch registers
	__m256 ymm0, ymm1, ymm2, ymm3;
	__m256 ymm4, ymm5, ymm6, ymm7;
	__m256 ymm8, ymm9, ymm10, ymm11;
	__m256 ymm12, ymm13, ymm14, ymm15;
	__m128 xmm0, xmm3;

	scomplex *tA = a;
	float *tAimag = &a->imag;
	scomplex *tB = b;
	scomplex *tC = c;
	for (n_iter = 0; n_iter < n0 / 8; n_iter++)
	{
		// clear scratch registers.
		ymm4 = _mm256_setzero_ps();
		ymm5 = _mm256_setzero_ps();
		ymm6 = _mm256_setzero_ps();
		ymm7 = _mm256_setzero_ps();
		ymm8 = _mm256_setzero_ps();
		ymm9 = _mm256_setzero_ps();
		ymm10 = _mm256_setzero_ps();
		ymm11 = _mm256_setzero_ps();
		ymm12 = _mm256_setzero_ps();
		ymm13 = _mm256_setzero_ps();
		ymm14 = _mm256_setzero_ps();
		ymm15 = _mm256_setzero_ps();

		dim_t ta_inc_row = rs_a;
		dim_t tb_inc_row = rs_b;
		dim_t tc_inc_row = rs_c;

		dim_t ta_inc_col = cs_a;
		dim_t tb_inc_col = cs_b;
		dim_t tc_inc_col = cs_c;

		tA = a;
		tAimag = &a->imag;
		tB = b + n_iter*tb_inc_col*8;
		tC = c + n_iter*tc_inc_col*8;
		for (k_iter = 0; k_iter <k0; k_iter++)
		{
			// The inner loop broadcasts the B matrix data and
			// multiplies it with the A matrix.
			// This loop is processing MR x K
			ymm0 = _mm256_loadu_ps((float const *)(tB + tb_inc_row * k_iter));
			ymm1 = _mm256_loadu_ps((float const *)(tB + tb_inc_row * k_iter +  4));

			//broadcasted matrix B elements are multiplied
			//with matrix A columns.
			ymm2 = _mm256_broadcast_ss((float const *)(tA));
			ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
			ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);

			ymm2 = _mm256_broadcast_ss((float const *)(tA + ta_inc_row));
			ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
			ymm9 = _mm256_fmadd_ps(ymm1, ymm2, ymm9);

			ymm2 = _mm256_broadcast_ss((float const *)(tA + ta_inc_row*2));
			ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
			ymm13 = _mm256_fmadd_ps(ymm1, ymm2, ymm13);

			//Compute imag values
			ymm2 = _mm256_broadcast_ss((float const *)(tAimag ));
			ymm6 = _mm256_fmadd_ps(ymm0, ymm2, ymm6);
			ymm7 = _mm256_fmadd_ps(ymm1, ymm2, ymm7);

			ymm2 = _mm256_broadcast_ss((float const *)(tAimag + ta_inc_row *2));
			ymm10 = _mm256_fmadd_ps(ymm0, ymm2, ymm10);
			ymm11 = _mm256_fmadd_ps(ymm1, ymm2, ymm11);

			ymm2 = _mm256_broadcast_ss((float const *)(tAimag + ta_inc_row *4));
			ymm14 = _mm256_fmadd_ps(ymm0, ymm2, ymm14);
			ymm15 = _mm256_fmadd_ps(ymm1, ymm2, ymm15);
			tA += ta_inc_col;
			tAimag += ta_inc_col*2;
		}
		ymm6 =_mm256_permute_ps(ymm6,  0xb1);
		ymm7 =_mm256_permute_ps(ymm7,  0xb1);
		ymm10 = _mm256_permute_ps(ymm10, 0xb1);
		ymm11 = _mm256_permute_ps(ymm11, 0xb1);
		ymm14 = _mm256_permute_ps(ymm14, 0xb1);
		ymm15 = _mm256_permute_ps(ymm15, 0xb1);

		// subtract/add even/odd elements
		ymm4 = _mm256_addsub_ps(ymm4, ymm6);
		ymm5 = _mm256_addsub_ps(ymm5, ymm7);

		ymm8 = _mm256_addsub_ps(ymm8, ymm10);
		ymm9 = _mm256_addsub_ps(ymm9, ymm11);

		ymm12 = _mm256_addsub_ps( ymm12, ymm14);
		ymm13 = _mm256_addsub_ps( ymm13, ymm15);

		// alpha, beta multiplication.

		/* (ar + ai) x AB */
		ymm0 = _mm256_broadcast_ss((float const *)(alpha));       // load alpha_r and duplicate
		ymm1 = _mm256_broadcast_ss((float const *)(&alpha->imag));    // load alpha_i and duplicate

		ymm3 = _mm256_permute_ps(ymm4, 0xb1);
		ymm4 = _mm256_mul_ps(ymm0, ymm4);
		ymm3 =_mm256_mul_ps(ymm1, ymm3);
		ymm4 = _mm256_addsub_ps(ymm4, ymm3);

		ymm3 = _mm256_permute_ps(ymm5, 0xb1);
		ymm5 = _mm256_mul_ps(ymm0, ymm5);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm5 = _mm256_addsub_ps(ymm5, ymm3);

		ymm3 = _mm256_permute_ps(ymm8, 0xb1);
		ymm8 = _mm256_mul_ps(ymm0, ymm8);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm8 = _mm256_addsub_ps(ymm8, ymm3);

		ymm3 = _mm256_permute_ps(ymm9, 0xb1);
		ymm9 = _mm256_mul_ps(ymm0, ymm9);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm9 = _mm256_addsub_ps(ymm9, ymm3);

		ymm3 = _mm256_permute_ps(ymm12, 0xb1);
		ymm12 = _mm256_mul_ps(ymm0, ymm12);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm12 = _mm256_addsub_ps(ymm12, ymm3);

		ymm3 = _mm256_permute_ps(ymm13, 0xb1);
		ymm13 = _mm256_mul_ps(ymm0, ymm13);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm13 = _mm256_addsub_ps(ymm13, ymm3);

		if(tc_inc_row == 1) //col stored
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				//transpose left 3x4
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm4), _mm256_castps_pd (ymm8)));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

				ymm1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd (ymm4) , _mm256_castps_pd(ymm8)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm1));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC ) ,_mm256_extractf128_ps (ymm0,1));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12, 1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm1,1));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12,1));

				//transpose right 3x4
				tC += tc_inc_col;
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm5), _mm256_castps_pd(ymm9)));
				_mm_storeu_ps((float *)(tC ),_mm256_castps256_ps128(ymm0));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm13));

				ymm1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(ymm5), _mm256_castps_pd(ymm9)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm1));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm13));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC  ),_mm256_extractf128_ps (ymm0,1));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm13,1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC  ),_mm256_extractf128_ps (ymm1,1));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm13,1));

			}
			else{
				ymm1 = _mm256_broadcast_ss((float const *)(beta));       // load alpha_r and duplicate
				ymm2 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load alpha_i and duplicate

				//Multiply ymm4 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC) );
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *) (tC + tc_inc_col*2));
				xmm3 = _mm_loadh_pi(xmm3,  (__m64 const *)(tC + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm4 = _mm256_add_ps(ymm4, ymm0);

				//Multiply ymm8 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 1)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 1 + tc_inc_col)) ;
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC + 1 + tc_inc_col*2)) ;
				xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC + 1 + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm8 = _mm256_add_ps(ymm8, ymm0);

				//Multiply ymm12 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 2)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 2 + tc_inc_col)) ;
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC + 2 + tc_inc_col*2)) ;
				xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC + 2 + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm12 = _mm256_add_ps(ymm12, ymm0);

				//transpose left 3x4
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm4), _mm256_castps_pd (ymm8)));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

				ymm3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd (ymm4) , _mm256_castps_pd(ymm8)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm3));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm0,1));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12, 1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC  ),_mm256_extractf128_ps (ymm3,1));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12,1));

					//Multiply ymm5 with beta
				tC += tc_inc_col;
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *) (tC + tc_inc_col*2));
				xmm3 = _mm_loadh_pi(xmm3,  (__m64 const *)(tC + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm5 = _mm256_add_ps(ymm5, ymm0);

				//Multiply ymm9 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC+ 1)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC+ 1 + tc_inc_col)) ;
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC+ 1 + tc_inc_col*2)) ;
				xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC+ 1 + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm9 = _mm256_add_ps(ymm9, ymm0);

				//Multiply ymm13 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 2)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 2 + tc_inc_col)) ;
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC + 2 + tc_inc_col*2)) ;
				xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC + 2 + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm13 = _mm256_add_ps(ymm13, ymm0);

				//transpose right 3x4
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm5), _mm256_castps_pd(ymm9)));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm13));

				ymm3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(ymm5), _mm256_castps_pd(ymm9)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC ), _mm256_castps256_ps128(ymm3));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm13));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC  ),_mm256_extractf128_ps (ymm0,1));
				_mm_storel_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm13,1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC  ),_mm256_extractf128_ps (ymm3,1));
				_mm_storeh_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm13,1));
			}
		}
		else
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				_mm256_storeu_ps((float*)(tC), ymm4);
				_mm256_storeu_ps((float*)(tC + 4), ymm5);
				_mm256_storeu_ps((float*)(tC + tc_inc_row ),  ymm8);
				_mm256_storeu_ps((float*)(tC + tc_inc_row + 4), ymm9);
				_mm256_storeu_ps((float*)(tC + tc_inc_row *2), ymm12);
				_mm256_storeu_ps((float*)(tC + tc_inc_row *2+ 4), ymm13);
			}
			else{
				/* (br + bi) C + (ar + ai) AB */
				ymm0 = _mm256_broadcast_ss((float const *)(beta));       // load beta_r and duplicate
				ymm1 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load beta_i and duplicate

				ymm2 = _mm256_loadu_ps((float const *)(tC));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 =_mm256_mul_ps(ymm1, ymm3);
				ymm4 = _mm256_add_ps(ymm4, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+4));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm5 = _mm256_add_ps(ymm5, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm8 = _mm256_add_ps(ymm8, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row + 4));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm9 = _mm256_add_ps(ymm9, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row*2));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm12 = _mm256_add_ps(ymm12, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row*2 +4));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm13 = _mm256_add_ps(ymm13, _mm256_addsub_ps(ymm2, ymm3));

				_mm256_storeu_ps((float*)(tC), ymm4);
				_mm256_storeu_ps((float*)(tC + 4), ymm5);
				_mm256_storeu_ps((float*)(tC + tc_inc_row) ,  ymm8);
				_mm256_storeu_ps((float*)(tC + tc_inc_row + 4), ymm9);
				_mm256_storeu_ps((float*)(tC + tc_inc_row *2), ymm12);
				_mm256_storeu_ps((float*)(tC + tc_inc_row *2+ 4), ymm13);
			}
		}
	}

	consider_edge_cases:
	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		scomplex* restrict cij = c + j_edge*cs_c;
		scomplex* restrict ai  = a;
		scomplex* restrict bj  = b + n_iter*8;

		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_cgemmsup_rv_zen_asm_3x4
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

			bli_cgemmsup_rv_zen_asm_3x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
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
	}

}

void bli_cgemmsup_rv_zen_asm_2x8n
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

	uint64_t k_iter = 0;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------
	//scratch registers
	__m256 ymm0, ymm1, ymm2, ymm3;
	__m256 ymm4, ymm5, ymm6, ymm7;
	__m256 ymm8, ymm9, ymm10, ymm11;
	__m128 xmm0, xmm3;

	scomplex *tA = a;
	float *tAimag = &a->imag;
	scomplex *tB = b;
	scomplex *tC = c;
	for (n_iter = 0; n_iter < n0 / 8; n_iter++)
	{
		// clear scratch registers.
		ymm4 = _mm256_setzero_ps();
		ymm5 = _mm256_setzero_ps();
		ymm6 = _mm256_setzero_ps();
		ymm7 = _mm256_setzero_ps();
		ymm8 = _mm256_setzero_ps();
		ymm9 = _mm256_setzero_ps();
		ymm10 = _mm256_setzero_ps();
		ymm11 = _mm256_setzero_ps();

		dim_t ta_inc_row = rs_a;
		dim_t tb_inc_row = rs_b;
		dim_t tc_inc_row = rs_c;

		dim_t ta_inc_col = cs_a;
		dim_t tb_inc_col = cs_b;
		dim_t tc_inc_col = cs_c;

		tA = a;
		tAimag = &a->imag;
		tB = b + n_iter*tb_inc_col*8;
		tC = c + n_iter*tc_inc_col*8;
		for (k_iter = 0; k_iter <k0; k_iter++)
		{
			// The inner loop broadcasts the B matrix data and
			// multiplies it with the A matrix.
			// This loop is processing MR x K
			ymm0 = _mm256_loadu_ps((float const *)(tB + tb_inc_row * k_iter));
			ymm1 = _mm256_loadu_ps((float const *)(tB + tb_inc_row * k_iter +  4));

			//broadcasted matrix B elements are multiplied
			//with matrix A columns.
			ymm2 = _mm256_broadcast_ss((float const *)(tA));
			ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
			ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);

			ymm2 = _mm256_broadcast_ss((float const *)(tA + ta_inc_row));
			ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
			ymm9 = _mm256_fmadd_ps(ymm1, ymm2, ymm9);

			//Compute imag values
			ymm2 = _mm256_broadcast_ss((float const *)(tAimag) );
			ymm6 = _mm256_fmadd_ps(ymm0, ymm2, ymm6);
			ymm7 = _mm256_fmadd_ps(ymm1, ymm2, ymm7);

			ymm2 = _mm256_broadcast_ss((float const *)(tAimag + ta_inc_row *2));
			ymm10 = _mm256_fmadd_ps(ymm0, ymm2, ymm10);
			ymm11 = _mm256_fmadd_ps(ymm1, ymm2, ymm11);

			tA += ta_inc_col;
			tAimag += ta_inc_col*2;
		}
		ymm6 =_mm256_permute_ps(ymm6,  0xb1);
		ymm7 =_mm256_permute_ps(ymm7,  0xb1);
		ymm10 = _mm256_permute_ps(ymm10, 0xb1);
		ymm11 = _mm256_permute_ps(ymm11, 0xb1);

		// subtract/add even/odd elements
		ymm4 = _mm256_addsub_ps(ymm4, ymm6);
		ymm5 = _mm256_addsub_ps(ymm5, ymm7);

		ymm8 = _mm256_addsub_ps(ymm8, ymm10);
		ymm9 = _mm256_addsub_ps(ymm9, ymm11);

		// alpha, beta multiplication.

		/* (ar + ai) x AB */
		ymm0 = _mm256_broadcast_ss((float const *)(alpha));       // load alpha_r and duplicate
		ymm1 = _mm256_broadcast_ss((float const *)(&alpha->imag));    // load alpha_i and duplicate

		ymm3 = _mm256_permute_ps(ymm4, 0xb1);
		ymm4 = _mm256_mul_ps(ymm0, ymm4);
		ymm3 =_mm256_mul_ps(ymm1, ymm3);
		ymm4 = _mm256_addsub_ps(ymm4, ymm3);

		ymm3 = _mm256_permute_ps(ymm5, 0xb1);
		ymm5 = _mm256_mul_ps(ymm0, ymm5);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm5 = _mm256_addsub_ps(ymm5, ymm3);

		ymm3 = _mm256_permute_ps(ymm8, 0xb1);
		ymm8 = _mm256_mul_ps(ymm0, ymm8);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm8 = _mm256_addsub_ps(ymm8, ymm3);

		ymm3 = _mm256_permute_ps(ymm9, 0xb1);
		ymm9 = _mm256_mul_ps(ymm0, ymm9);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm9 = _mm256_addsub_ps(ymm9, ymm3);

		if(tc_inc_row == 1) //col stored
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				//transpose left 2x4
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm4), _mm256_castps_pd (ymm8)));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));

				ymm1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd (ymm4) , _mm256_castps_pd(ymm8)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm0,1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm1,1));

				//transpose right 2x4
				tC += tc_inc_col;
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm5), _mm256_castps_pd(ymm9)));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));

				ymm1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(ymm5), _mm256_castps_pd(ymm9)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm0,1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm1,1));

			}
			else{
				ymm1 = _mm256_broadcast_ss((float const *)(beta));       // load alpha_r and duplicate
				ymm2 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load alpha_i and duplicate

				//Multiply ymm4 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *) (tC + tc_inc_col*2));
				xmm3 = _mm_loadh_pi(xmm3,  (__m64 const *)(tC + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm4 = _mm256_add_ps(ymm4, ymm0);

				//Multiply ymm8 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 1)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 1 + tc_inc_col)) ;
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC + 1 + tc_inc_col*2)) ;
				xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC + 1 + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1);
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm8 = _mm256_add_ps(ymm8, ymm0);

				//transpose left 2x4
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm4), _mm256_castps_pd (ymm8)));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));

				ymm3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd (ymm4) , _mm256_castps_pd(ymm8)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm3));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm0,1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm3,1));

				//Multiply ymm5 with beta
				tC += tc_inc_col;
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *) (tC + tc_inc_col*2));
				xmm3 = _mm_loadh_pi(xmm3,  (__m64 const *)(tC + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm5 = _mm256_add_ps(ymm5, ymm0);

				//Multiply ymm9 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC+ 1)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC+ 1 + tc_inc_col)) ;
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC+ 1 + tc_inc_col*2)) ;
				xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC+ 1 + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm9 = _mm256_add_ps(ymm9, ymm0);

				//transpose right 2x4
				ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm5), _mm256_castps_pd(ymm9)));
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));

				ymm3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(ymm5), _mm256_castps_pd(ymm9)));
				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm3));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm0,1));

				tC += tc_inc_col;
				_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm3,1));

			}
		}
		else
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				_mm256_storeu_ps((float*)(tC), ymm4);
				_mm256_storeu_ps((float*)(tC + 4), ymm5);
				_mm256_storeu_ps((float*)(tC + tc_inc_row) ,  ymm8);
				_mm256_storeu_ps((float*)(tC + tc_inc_row + 4), ymm9);
			}
			else{
				/* (br + bi) C + (ar + ai) AB */
				ymm0 = _mm256_broadcast_ss((float const *)(beta));       // load beta_r and duplicate
				ymm1 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load beta_i and duplicate

				ymm2 = _mm256_loadu_ps((float const *)(tC));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 =_mm256_mul_ps(ymm1, ymm3);
				ymm4 = _mm256_add_ps(ymm4, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+4));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm5 = _mm256_add_ps(ymm5, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm8 = _mm256_add_ps(ymm8, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row + 4));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm9 = _mm256_add_ps(ymm9, _mm256_addsub_ps(ymm2, ymm3));

				_mm256_storeu_ps((float*)(tC), ymm4);
				_mm256_storeu_ps((float*)(tC + 4), ymm5);
				_mm256_storeu_ps((float*)(tC + tc_inc_row) ,  ymm8);
				_mm256_storeu_ps((float*)(tC + tc_inc_row + 4), ymm9);
			}
		}
	}
		consider_edge_cases:
	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		scomplex* restrict cij = c + j_edge*cs_c;
		scomplex* restrict ai  = a;
		scomplex* restrict bj  = b + n_iter * 8 ;

		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_cgemmsup_rv_zen_asm_2x4
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

			bli_cgemmsup_rv_zen_asm_2x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
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
	}
}

void bli_cgemmsup_rv_zen_asm_1x8n
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

	uint64_t k_iter = 0;

	uint64_t n_iter = n0 / 8;
	uint64_t n_left = n0 % 8;

	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------
	//scratch registers
	__m256 ymm0, ymm1, ymm2, ymm3;
	__m256 ymm4, ymm5, ymm6, ymm7;
	__m128 xmm0, xmm3;

	scomplex *tA = a;
	float *tAimag = &a->imag;
	scomplex *tB = b;
	scomplex *tC = c;
	for (n_iter = 0; n_iter < n0 / 8; n_iter++)
	{
		// clear scratch registers.
		ymm4 = _mm256_setzero_ps();
		ymm5 = _mm256_setzero_ps();
		ymm6 = _mm256_setzero_ps();
		ymm7 = _mm256_setzero_ps();

		dim_t tb_inc_row = rs_b;
		dim_t tc_inc_row = rs_c;

		dim_t ta_inc_col = cs_a;
		dim_t tb_inc_col = cs_b;
		dim_t tc_inc_col = cs_c;

		tA = a;
		tAimag = &a->imag;
		tB = b + n_iter*tb_inc_col*8;
		tC = c + n_iter*tc_inc_col*8;
		for (k_iter = 0; k_iter <k0; k_iter++)
		{
			// The inner loop broadcasts the B matrix data and
			// multiplies it with the A matrix.
			// This loop is processing MR x K
			ymm0 = _mm256_loadu_ps((float const *)(tB + tb_inc_row * k_iter));
			ymm1 = _mm256_loadu_ps((float const *)(tB + tb_inc_row * k_iter +  4));

			//broadcasted matrix B elements are multiplied
			//with matrix A columns.
			ymm2 = _mm256_broadcast_ss((float const *)(tA));
			ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
			ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);

			//Compute imag values
			ymm2 = _mm256_broadcast_ss((float const *)(tAimag) );
			ymm6 = _mm256_fmadd_ps(ymm0, ymm2, ymm6);
			ymm7 = _mm256_fmadd_ps(ymm1, ymm2, ymm7);

			tA += ta_inc_col;
			tAimag += ta_inc_col*2;
		}
		ymm6 =_mm256_permute_ps(ymm6,  0xb1);
		ymm7 =_mm256_permute_ps(ymm7,  0xb1);

		// subtract/add even/odd elements
		ymm4 = _mm256_addsub_ps(ymm4, ymm6);
		ymm5 = _mm256_addsub_ps(ymm5, ymm7);

		// alpha, beta multiplication.

		/* (ar + ai) x AB */
		ymm0 = _mm256_broadcast_ss((float const *)(alpha));       // load alpha_r and duplicate
		ymm1 = _mm256_broadcast_ss((float const *)(&alpha->imag));    // load alpha_i and duplicate

		ymm3 = _mm256_permute_ps(ymm4, 0xb1);
		ymm4 = _mm256_mul_ps(ymm0, ymm4);
		ymm3 =_mm256_mul_ps(ymm1, ymm3);
		ymm4 = _mm256_addsub_ps(ymm4, ymm3);

		ymm3 = _mm256_permute_ps(ymm5, 0xb1);
		ymm5 = _mm256_mul_ps(ymm0, ymm5);
		ymm3 = _mm256_mul_ps(ymm1, ymm3);
		ymm5 = _mm256_addsub_ps(ymm5, ymm3);

		if(tc_inc_row == 1) //col stored
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				//transpose left 1x4
				_mm_storel_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm4));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm4));

				tC += tc_inc_col;
				_mm_storel_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm4,1));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm4,1));

				//transpose right 1x4
				tC += tc_inc_col;
				_mm_storel_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm5));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm5));

				tC += tc_inc_col;
				_mm_storel_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm5,1));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm5,1));

			}
			else{
				ymm1 = _mm256_broadcast_ss((float const *)(beta));       // load alpha_r and duplicate
				ymm2 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load alpha_i and duplicate

				//Multiply ymm4 with beta
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *) (tC + tc_inc_col*2));
				xmm3 = _mm_loadh_pi(xmm3,  (__m64 const *)(tC + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm4 = _mm256_add_ps(ymm4, ymm0);

				_mm_storel_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm4));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm4));

				tC += tc_inc_col;
				_mm_storel_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm4,1));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm4,1));

				//Multiply ymm5 with beta
				tC += tc_inc_col;
				xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC)) ;
				xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
				xmm3 = _mm_loadl_pi(xmm3, (__m64 const *) (tC + tc_inc_col*2));
				xmm3 = _mm_loadh_pi(xmm3,  (__m64 const *)(tC + tc_inc_col*3)) ;
				ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_ps(ymm0, 0xb1);
				ymm0 = _mm256_mul_ps(ymm1, ymm0);
				ymm3 = _mm256_mul_ps(ymm2, ymm3);
				ymm0 = _mm256_addsub_ps(ymm0, ymm3);
				ymm5 = _mm256_add_ps(ymm5, ymm0);

				_mm_storel_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm5));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC), _mm256_castps256_ps128(ymm5));

				tC += tc_inc_col;
				_mm_storel_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm5,1));

				tC += tc_inc_col;
				_mm_storeh_pi((__m64 *)(tC)  ,_mm256_extractf128_ps (ymm5,1));

			}
		}
		else
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				_mm256_storeu_ps((float*)(tC), ymm4);
				_mm256_storeu_ps((float*)(tC + 4), ymm5);
			}
			else{
				/* (br + bi) C + (ar + ai) AB */
				ymm0 = _mm256_broadcast_ss((float const *)(beta));       // load beta_r and duplicate
				ymm1 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load beta_i and duplicate

				ymm2 = _mm256_loadu_ps((float const *)(tC));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 =_mm256_mul_ps(ymm1, ymm3);
				ymm4 = _mm256_add_ps(ymm4, _mm256_addsub_ps(ymm2, ymm3));

				ymm2 = _mm256_loadu_ps((float const *)(tC+4));
				ymm3 = _mm256_permute_ps(ymm2, 0xb1);
				ymm2 = _mm256_mul_ps(ymm0, ymm2);
				ymm3 = _mm256_mul_ps(ymm1, ymm3);
				ymm5 = _mm256_add_ps(ymm5, _mm256_addsub_ps(ymm2, ymm3));

				_mm256_storeu_ps((float*)(tC), ymm4);
				_mm256_storeu_ps((float*)(tC + 4), ymm5);
			}
		}
	}
		consider_edge_cases:
	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		scomplex* restrict cij = c + j_edge*cs_c;
		scomplex* restrict ai  = a;
		scomplex* restrict bj  = b + n_iter * 8;

		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_cgemmsup_rv_zen_asm_1x4
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

			bli_cgemmsup_rv_zen_asm_1x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left ){
			bli_cgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
		}
	}
}


void bli_cgemmsup_rv_zen_asm_3x4
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

	uint64_t k_iter = 0;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	// -------------------------------------------------------------------------
	//scratch registers
	__m256 ymm0, ymm1, ymm2, ymm3;
	__m256 ymm4, ymm6;
	__m256 ymm8, ymm10;
	__m256 ymm12, ymm14;
	__m128 xmm0, xmm3;

	scomplex *tA = a;
	float *tAimag = &a->imag;
	scomplex *tB = b;
	scomplex *tC = c;
	// clear scratch registers.
	ymm4 = _mm256_setzero_ps();
	ymm6 = _mm256_setzero_ps();
	ymm8 = _mm256_setzero_ps();
	ymm10 = _mm256_setzero_ps();
	ymm12 = _mm256_setzero_ps();
	ymm14 = _mm256_setzero_ps();

	dim_t ta_inc_row = rs_a;
	dim_t tb_inc_row = rs_b;
	dim_t tc_inc_row = rs_c;

	dim_t ta_inc_col = cs_a;
	dim_t tc_inc_col = cs_c;

	for (k_iter = 0; k_iter <k0; k_iter++)
	{
		// The inner loop broadcasts the B matrix data and
		// multiplies it with the A matrix.
		// This loop is processing MR x K
		ymm0 = _mm256_loadu_ps((float const *)(tB + tb_inc_row * k_iter));

		//broadcasted matrix B elements are multiplied
		//with matrix A columns.
		ymm2 = _mm256_broadcast_ss((float const *)(tA));
		ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);

		ymm2 = _mm256_broadcast_ss((float const *)(tA + ta_inc_row));
		ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);

		ymm2 = _mm256_broadcast_ss((float const *)(tA + ta_inc_row*2));
		ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);

		//Compute imag values
		ymm2 = _mm256_broadcast_ss((float const *)(tAimag ));
		ymm6 = _mm256_fmadd_ps(ymm0, ymm2, ymm6);

		ymm2 = _mm256_broadcast_ss((float const *)(tAimag + ta_inc_row *2));
		ymm10 = _mm256_fmadd_ps(ymm0, ymm2, ymm10);

		ymm2 = _mm256_broadcast_ss((float const *)(tAimag + ta_inc_row *4));
		ymm14 = _mm256_fmadd_ps(ymm0, ymm2, ymm14);
		tA += ta_inc_col;
		tAimag += ta_inc_col*2;
	}
	ymm6 =_mm256_permute_ps(ymm6,  0xb1);
	ymm10 = _mm256_permute_ps(ymm10, 0xb1);
	ymm14 = _mm256_permute_ps(ymm14, 0xb1);

	// subtract/add even/odd elements
	ymm4 = _mm256_addsub_ps(ymm4, ymm6);

	ymm8 = _mm256_addsub_ps(ymm8, ymm10);

	ymm12 = _mm256_addsub_ps( ymm12, ymm14);

	// alpha, beta multiplication.

	/* (ar + ai) x AB */
	ymm0 = _mm256_broadcast_ss((float const *)(alpha));       // load alpha_r and duplicate
	ymm1 = _mm256_broadcast_ss((float const *)(&alpha->imag));    // load alpha_i and duplicate

	ymm3 = _mm256_permute_ps(ymm4, 0xb1);
	ymm4 = _mm256_mul_ps(ymm0, ymm4);
	ymm3 =_mm256_mul_ps(ymm1, ymm3);
	ymm4 = _mm256_addsub_ps(ymm4, ymm3);

	ymm3 = _mm256_permute_ps(ymm8, 0xb1);
	ymm8 = _mm256_mul_ps(ymm0, ymm8);
	ymm3 = _mm256_mul_ps(ymm1, ymm3);
	ymm8 = _mm256_addsub_ps(ymm8, ymm3);

	ymm3 = _mm256_permute_ps(ymm12, 0xb1);
	ymm12 = _mm256_mul_ps(ymm0, ymm12);
	ymm3 = _mm256_mul_ps(ymm1, ymm3);
	ymm12 = _mm256_addsub_ps(ymm12, ymm3);

	if(tc_inc_row == 1) //col stored
	{
		if(beta->real == 0.0 && beta->imag == 0.0)
		{
			//transpose 3x4
			ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm4), _mm256_castps_pd (ymm8)));
			_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));
			_mm_storel_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

			ymm1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd (ymm4) , _mm256_castps_pd(ymm8)));
			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm1));
			_mm_storeh_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC),_mm256_extractf128_ps (ymm0,1));
			_mm_storel_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12, 1));

			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC ) ,_mm256_extractf128_ps (ymm1,1));
			_mm_storeh_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12,1));

		}
		else{
			ymm1 = _mm256_broadcast_ss((float const *)(beta));       // load alpha_r and duplicate
			ymm2 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load alpha_i and duplicate

			//Multiply ymm4 with beta
			xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC)) ;
			xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
			xmm3 = _mm_loadl_pi(xmm3, (__m64 const *) (tC + tc_inc_col*2));
			xmm3 = _mm_loadh_pi(xmm3,  (__m64 const *)(tC + tc_inc_col*3)) ;
			ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
			ymm3 = _mm256_permute_ps(ymm0, 0xb1);
			ymm0 = _mm256_mul_ps(ymm1, ymm0);
			ymm3 = _mm256_mul_ps(ymm2, ymm3);
			ymm0 = _mm256_addsub_ps(ymm0, ymm3);
			ymm4 = _mm256_add_ps(ymm4, ymm0);

			//Multiply ymm8 with beta
			xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 1)) ;
			xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 1 + tc_inc_col)) ;
			xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC + 1 + tc_inc_col*2)) ;
			xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC + 1 + tc_inc_col*3)) ;
			ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
			ymm3 = _mm256_permute_ps(ymm0, 0xb1);
			ymm0 = _mm256_mul_ps(ymm1, ymm0);
			ymm3 = _mm256_mul_ps(ymm2, ymm3);
			ymm0 = _mm256_addsub_ps(ymm0, ymm3);
			ymm8 = _mm256_add_ps(ymm8, ymm0);

			//Multiply ymm12 with beta
			xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 2)) ;
			xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 2 + tc_inc_col)) ;
			xmm3 = _mm_loadl_pi(xmm3, (__m64 const *)(tC + 2 + tc_inc_col*2)) ;
			xmm3 = _mm_loadh_pi(xmm3, (__m64 const *)(tC + 2 + tc_inc_col*3)) ;
			ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm3, 1) ;
			ymm3 = _mm256_permute_ps(ymm0, 0xb1);
			ymm0 = _mm256_mul_ps(ymm1, ymm0);
			ymm3 = _mm256_mul_ps(ymm2, ymm3);
			ymm0 = _mm256_addsub_ps(ymm0, ymm3);
			ymm12 = _mm256_add_ps(ymm12, ymm0);

			//transpose 3x4
			ymm0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd (ymm4), _mm256_castps_pd (ymm8)));
			_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm0));
			_mm_storel_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

			ymm3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd (ymm4) , _mm256_castps_pd(ymm8)));
			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC), _mm256_castps256_ps128(ymm3));
			_mm_storeh_pi((__m64 *)(tC+2), _mm256_castps256_ps128(ymm12));

			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC)  ,_mm256_extractf128_ps (ymm0,1));
			_mm_storel_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12, 1));

			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC  ),_mm256_extractf128_ps (ymm3,1));
			_mm_storeh_pi((__m64 *)(tC+2), _mm256_extractf128_ps(ymm12,1));
		}
	}
	else
	{
		if(beta->real == 0.0 && beta->imag == 0.0)
		{
			_mm256_storeu_ps((float*)(tC), ymm4);
			_mm256_storeu_ps((float*)(tC + tc_inc_row) ,  ymm8);
			_mm256_storeu_ps((float*)(tC + tc_inc_row *2), ymm12);
		}
		else{
			/* (br + bi) C + (ar + ai) AB */
			ymm0 = _mm256_broadcast_ss((float const *)(beta));       // load beta_r and duplicate
			ymm1 = _mm256_broadcast_ss((float const *)(&beta->imag));    // load beta_i and duplicate

			ymm2 = _mm256_loadu_ps((float const *)(tC));
			ymm3 = _mm256_permute_ps(ymm2, 0xb1);
			ymm2 = _mm256_mul_ps(ymm0, ymm2);
			ymm3 =_mm256_mul_ps(ymm1, ymm3);
			ymm4 = _mm256_add_ps(ymm4, _mm256_addsub_ps(ymm2, ymm3));

			ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row));
			ymm3 = _mm256_permute_ps(ymm2, 0xb1);
			ymm2 = _mm256_mul_ps(ymm0, ymm2);
			ymm3 = _mm256_mul_ps(ymm1, ymm3);
			ymm8 = _mm256_add_ps(ymm8, _mm256_addsub_ps(ymm2, ymm3));

			ymm2 = _mm256_loadu_ps((float const *)(tC+tc_inc_row*2));
			ymm3 = _mm256_permute_ps(ymm2, 0xb1);
			ymm2 = _mm256_mul_ps(ymm0, ymm2);
			ymm3 = _mm256_mul_ps(ymm1, ymm3);
			ymm12 = _mm256_add_ps(ymm12, _mm256_addsub_ps(ymm2, ymm3));

			_mm256_storeu_ps((float*)(tC), ymm4);
			_mm256_storeu_ps((float*)(tC + tc_inc_row) ,  ymm8);
			_mm256_storeu_ps((float*)(tC + tc_inc_row *2), ymm12);;
		}
	}
}

void bli_cgemmsup_rv_zen_asm_3x2
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

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = 0;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	scomplex *tA = a;
	float *tAimag = &a->imag;
	scomplex *tB = b;
	scomplex *tC = c;
	// clear scratch registers.
	__m128 xmm0, xmm1, xmm2, xmm3;
	__m128 xmm4 = _mm_setzero_ps();
	__m128 xmm6 = _mm_setzero_ps();
	__m128 xmm8 = _mm_setzero_ps();
	__m128 xmm10 = _mm_setzero_ps();
	__m128 xmm12 = _mm_setzero_ps();
	__m128 xmm14 = _mm_setzero_ps();

	dim_t ta_inc_row = rs_a;
	dim_t tb_inc_row = rs_b;
	dim_t tc_inc_row = rs_c;

	dim_t ta_inc_col = cs_a;
	dim_t tc_inc_col = cs_c;

	for (k_iter = 0; k_iter <k0; k_iter++)
	{
		// The inner loop broadcasts the B matrix data and
		// multiplies it with the A matrix.
		// This loop is processing MR x K
		xmm0 = _mm_loadu_ps((float const *)(tB + tb_inc_row * k_iter));

		//broadcasted matrix B elements are multiplied
		//with matrix A columns.
		xmm2 = _mm_broadcast_ss((float const *)(tA));
		xmm4 = _mm_fmadd_ps(xmm0, xmm2, xmm4);

		xmm2 = _mm_broadcast_ss((float const *)(tA + ta_inc_row));
		xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);

		xmm2 = _mm_broadcast_ss((float const *)(tA + ta_inc_row*2));
		xmm12 = _mm_fmadd_ps(xmm0, xmm2, xmm12);

		//Compute imag values
		xmm2 = _mm_broadcast_ss((float const *)(tAimag ));
		xmm6 = _mm_fmadd_ps(xmm0, xmm2, xmm6);

		xmm2 = _mm_broadcast_ss((float const *)(tAimag + ta_inc_row *2));
		xmm10 = _mm_fmadd_ps(xmm0, xmm2, xmm10);

		xmm2 = _mm_broadcast_ss((float const *)(tAimag + ta_inc_row *4));
		xmm14 = _mm_fmadd_ps(xmm0, xmm2, xmm14);
		tA += ta_inc_col;
		tAimag += ta_inc_col*2;
	}
	xmm6 =_mm_permute_ps(xmm6,  0xb1);
	xmm10 = _mm_permute_ps(xmm10, 0xb1);
	xmm14 = _mm_permute_ps(xmm14, 0xb1);

	// subtract/add even/odd elements
	xmm4 = _mm_addsub_ps(xmm4, xmm6);

	xmm8 = _mm_addsub_ps(xmm8, xmm10);

	xmm12 = _mm_addsub_ps( xmm12, xmm14);

	// alpha, beta multiplication.

	/* (ar + ai) x AB */
	xmm0 = _mm_broadcast_ss((float const *)(alpha));       // load alpha_r and duplicate
	xmm1 = _mm_broadcast_ss((float const *)(&alpha->imag));    // load alpha_i and duplicate

	xmm3 = _mm_permute_ps(xmm4, 0xb1);
	xmm4 = _mm_mul_ps(xmm0, xmm4);
	xmm3 =_mm_mul_ps(xmm1, xmm3);
	xmm4 = _mm_addsub_ps(xmm4, xmm3);

	xmm3 = _mm_permute_ps(xmm8, 0xb1);
	xmm8 = _mm_mul_ps(xmm0, xmm8);
	xmm3 = _mm_mul_ps(xmm1, xmm3);
	xmm8 = _mm_addsub_ps(xmm8, xmm3);

	xmm3 = _mm_permute_ps(xmm12, 0xb1);
	xmm12 = _mm_mul_ps(xmm0, xmm12);
	xmm3 = _mm_mul_ps(xmm1, xmm3);
	xmm12 = _mm_addsub_ps(xmm12, xmm3);

	if(tc_inc_row == 1) //col stored
	{
		if(beta->real == 0.0 && beta->imag == 0.0)
		{
			//transpose 3x2
			 xmm0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd (xmm4), _mm_castps_pd (xmm8)));
			_mm_storeu_ps((float *)(tC  ), xmm0);
			_mm_storel_pi((__m64 *)(tC+2), xmm12);

			xmm1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd (xmm4) , _mm_castps_pd(xmm8)));
			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC  ), xmm1);
			_mm_storeh_pi((__m64 *)(tC+2), xmm12);
		}
		else{
			xmm1 = _mm_broadcast_ss((float const *)(beta));       // load alpha_r and duplicate
			xmm2 = _mm_broadcast_ss((float const *)(&beta->imag));    // load alpha_i and duplicate

			//Multiply xmm4 with beta
			xmm0 = _mm_loadl_pi(xmm0, (__m64 const *) (tC)) ;
			xmm0 = _mm_loadh_pi(xmm0, (__m64 const *) (tC + tc_inc_col));
			xmm3 = _mm_permute_ps(xmm0, 0xb1);
			xmm0 = _mm_mul_ps(xmm1, xmm0);
			xmm3 = _mm_mul_ps(xmm2, xmm3);
			xmm0 = _mm_addsub_ps(xmm0, xmm3);
			xmm4 = _mm_add_ps(xmm4, xmm0);

			//Multiply xmm8 with beta
			xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 1)) ;
			xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 1 + tc_inc_col)) ;
			xmm3 = _mm_permute_ps(xmm0, 0xb1);
			xmm0 = _mm_mul_ps(xmm1, xmm0);
			xmm3 = _mm_mul_ps(xmm2, xmm3);
			xmm0 = _mm_addsub_ps(xmm0, xmm3);
			xmm8 = _mm_add_ps(xmm8, xmm0);

			//Multiply xmm12 with beta
			xmm0 = _mm_loadl_pi(xmm0, (__m64 const *)(tC + 2)) ;
			xmm0 = _mm_loadh_pi(xmm0, (__m64 const *)(tC + 2 + tc_inc_col)) ;
			xmm3 = _mm_permute_ps(xmm0, 0xb1);
			xmm0 = _mm_mul_ps(xmm1, xmm0);
			xmm3 = _mm_mul_ps(xmm2, xmm3);
			xmm0 = _mm_addsub_ps(xmm0, xmm3);
			xmm12 = _mm_add_ps(xmm12, xmm0);

			//transpose  3x2
			xmm0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd (xmm4), _mm_castps_pd (xmm8)));
			_mm_storeu_ps((float *)(tC  ), xmm0);
			_mm_storel_pi((__m64 *)(tC+2), xmm12);

			xmm3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd (xmm4) , _mm_castps_pd(xmm8)));
			tC += tc_inc_col;
			_mm_storeu_ps((float *)(tC  ), xmm3);
			_mm_storeh_pi((__m64 *)(tC+2), xmm12);

		}
	}
	else
	{
		if(beta->real == 0.0 && beta->imag == 0.0)
		{
			_mm_storeu_ps((float *)(tC), xmm4);
			_mm_storeu_ps((float *)(tC + tc_inc_row) ,  xmm8);
			_mm_storeu_ps((float *)(tC + tc_inc_row *2), xmm12);
		}
		else{
			/* (br + bi) C + (ar + ai) AB */
			xmm0 = _mm_broadcast_ss((float const *)(beta));       // load beta_r and duplicate
			xmm1 = _mm_broadcast_ss((float const *)(&beta->imag));    // load beta_i and duplicate

			xmm2 = _mm_loadu_ps((float const *)(tC));
			xmm3 = _mm_permute_ps(xmm2, 0xb1);
			xmm2 = _mm_mul_ps(xmm0, xmm2);
			xmm3 = _mm_mul_ps(xmm1, xmm3);
			xmm4 = _mm_add_ps(xmm4, _mm_addsub_ps(xmm2, xmm3));

			xmm2 = _mm_loadu_ps((float const *)(tC+tc_inc_row));
			xmm3 = _mm_permute_ps(xmm2, 0xb1);
			xmm2 = _mm_mul_ps(xmm0, xmm2);
			xmm3 = _mm_mul_ps(xmm1, xmm3);
			xmm8 = _mm_add_ps(xmm8, _mm_addsub_ps(xmm2, xmm3));

			xmm2 = _mm_loadu_ps((float const *)(tC+tc_inc_row*2));
			xmm3 = _mm_permute_ps(xmm2, 0xb1);
			xmm2 = _mm_mul_ps(xmm0, xmm2);
			xmm3 = _mm_mul_ps(xmm1, xmm3);
			xmm12 = _mm_add_ps(xmm12, _mm_addsub_ps(xmm2, xmm3));

			_mm_storeu_ps((float *)(tC), xmm4);
			_mm_storeu_ps((float *)(tC + tc_inc_row) ,  xmm8);
			_mm_storeu_ps((float *)(tC + tc_inc_row *2), xmm12);;
		}
	}
}
