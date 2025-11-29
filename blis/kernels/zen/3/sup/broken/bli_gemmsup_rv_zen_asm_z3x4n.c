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
void bli_zgemmsup_rv_zen_asm_3x4n
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
	uint64_t m_left = m0 % 3;
	if ( m_left )
	{
		zgemmsup_ker_ft ker_fps[3] =
		{
			NULL,
			bli_zgemmsup_rv_zen_asm_1x4n,
			bli_zgemmsup_rv_zen_asm_2x4n,
		};
		zgemmsup_ker_ft ker_fp = ker_fps[ m_left ];
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

	uint64_t k_iter = 0;


	uint64_t n_iter = n0 / 4;
	uint64_t n_left = n0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------
	//scratch registers
	__m256d ymm0, ymm1, ymm2, ymm3;
	__m256d ymm4, ymm5, ymm6, ymm7;
	__m256d ymm8, ymm9, ymm10, ymm11;
	__m256d ymm12, ymm13, ymm14, ymm15;
	__m128d xmm0, xmm3;

	dcomplex *tA = a;
	double *tAimag = &a->imag;
	dcomplex *tB = b;
	dcomplex *tC = c;
	for (n_iter = 0; n_iter < n0 / 4; n_iter++)
	{
		// clear scratch registers.
		ymm4 = _mm256_setzero_pd();
		ymm5 = _mm256_setzero_pd();
		ymm6 = _mm256_setzero_pd();
		ymm7 = _mm256_setzero_pd();
		ymm8 = _mm256_setzero_pd();
		ymm9 = _mm256_setzero_pd();
		ymm10 = _mm256_setzero_pd();
		ymm11 = _mm256_setzero_pd();
		ymm12 = _mm256_setzero_pd();
		ymm13 = _mm256_setzero_pd();
		ymm14 = _mm256_setzero_pd();
		ymm15 = _mm256_setzero_pd();

		dim_t ta_inc_row = rs_a;
		dim_t tb_inc_row = rs_b;
		dim_t tc_inc_row = rs_c;

		dim_t ta_inc_col = cs_a;
		dim_t tb_inc_col = cs_b;
		dim_t tc_inc_col = cs_c;

		tA = a;
		tAimag = &a->imag;
		tB = b + n_iter*tb_inc_col*4;
		tC = c + n_iter*tc_inc_col*4;
		for (k_iter = 0; k_iter <k0; k_iter++)
		{
			// The inner loop broadcasts the B matrix data and
			// multiplies it with the A matrix.
			// This loop is processing MR x K
			ymm0 = _mm256_loadu_pd((double const *)(tB + tb_inc_row * k_iter));
			ymm1 = _mm256_loadu_pd((double const *)(tB + tb_inc_row * k_iter +  2));

			//broadcasted matrix B elements are multiplied
			//with matrix A columns.
			ymm2 = _mm256_broadcast_sd((double const *)(tA));
			ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
			ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);

			ymm2 = _mm256_broadcast_sd((double const *)(tA + ta_inc_row));
			ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);
			ymm9 = _mm256_fmadd_pd(ymm1, ymm2, ymm9);

			ymm2 = _mm256_broadcast_sd((double const *)(tA + ta_inc_row*2));
			ymm12 = _mm256_fmadd_pd(ymm0, ymm2, ymm12);
			ymm13 = _mm256_fmadd_pd(ymm1, ymm2, ymm13);

			//Compute imag values
			ymm2 = _mm256_broadcast_sd((double const *)(tAimag ));
			ymm6 = _mm256_fmadd_pd(ymm0, ymm2, ymm6);
			ymm7 = _mm256_fmadd_pd(ymm1, ymm2, ymm7);

			ymm2 = _mm256_broadcast_sd((double const *)(tAimag + ta_inc_row *2));
			ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);
			ymm11 = _mm256_fmadd_pd(ymm1, ymm2, ymm11);

			ymm2 = _mm256_broadcast_sd((double const *)(tAimag + ta_inc_row *4));
			ymm14 = _mm256_fmadd_pd(ymm0, ymm2, ymm14);
			ymm15 = _mm256_fmadd_pd(ymm1, ymm2, ymm15);
			tA += ta_inc_col;
			tAimag += ta_inc_col*2;
		}
		ymm6 =_mm256_permute_pd(ymm6,  5);
		ymm7 =_mm256_permute_pd(ymm7,  5);
		ymm10 = _mm256_permute_pd(ymm10, 5);
		ymm11 = _mm256_permute_pd(ymm11, 5);
		ymm14 = _mm256_permute_pd(ymm14, 5);
		ymm15 = _mm256_permute_pd(ymm15, 5);

		// subtract/add even/odd elements
		ymm4 = _mm256_addsub_pd(ymm4, ymm6);
		ymm5 = _mm256_addsub_pd(ymm5, ymm7);

		ymm8 = _mm256_addsub_pd(ymm8, ymm10);
		ymm9 = _mm256_addsub_pd(ymm9, ymm11);

		ymm12 = _mm256_addsub_pd( ymm12, ymm14);
		ymm13 = _mm256_addsub_pd( ymm13, ymm15);

		// alpha, beta multiplication.

		/* (ar + ai) x AB */
		ymm0 = _mm256_broadcast_sd((double const *)(alpha));       // load alpha_r and duplicate
		ymm1 = _mm256_broadcast_sd((double const *)(&alpha->imag));    // load alpha_i and duplicate

		ymm3 = _mm256_permute_pd(ymm4, 5);
		ymm4 = _mm256_mul_pd(ymm0, ymm4);
		ymm3 =_mm256_mul_pd(ymm1, ymm3);
		ymm4 = _mm256_addsub_pd(ymm4, ymm3);

		ymm3 = _mm256_permute_pd(ymm5, 5);
		ymm5 = _mm256_mul_pd(ymm0, ymm5);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm5 = _mm256_addsub_pd(ymm5, ymm3);

		ymm3 = _mm256_permute_pd(ymm8, 5);
		ymm8 = _mm256_mul_pd(ymm0, ymm8);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm8 = _mm256_addsub_pd(ymm8, ymm3);

		ymm3 = _mm256_permute_pd(ymm9, 5);
		ymm9 = _mm256_mul_pd(ymm0, ymm9);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm9 = _mm256_addsub_pd(ymm9, ymm3);

		ymm3 = _mm256_permute_pd(ymm12, 5);
		ymm12 = _mm256_mul_pd(ymm0, ymm12);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm12 = _mm256_addsub_pd(ymm12, ymm3);

		ymm3 = _mm256_permute_pd(ymm13, 5);
		ymm13 = _mm256_mul_pd(ymm0, ymm13);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm13 = _mm256_addsub_pd(ymm13, ymm3);

		if(tc_inc_row == 1) //col stored
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				//transpose left 3x2
				_mm_storeu_pd((double *)(tC ), _mm256_castpd256_pd128(ymm4));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm8));
				_mm_storeu_pd((double *)(tC+2), _mm256_castpd256_pd128(ymm12));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm4,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm8,1));
				_mm_storeu_pd((double *)(tC+2), _mm256_extractf128_pd(ymm12, 1));
				tC += tc_inc_col;

				//transpose right 3x2
				_mm_storeu_pd((double *)(tC ), _mm256_castpd256_pd128(ymm5));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm9));
				_mm_storeu_pd((double *)(tC+2), _mm256_castpd256_pd128(ymm13));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm5,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm9,1));
				_mm_storeu_pd((double *)(tC+2), _mm256_extractf128_pd(ymm13, 1));
			}
			else{
				ymm1 = _mm256_broadcast_sd((double const *)(beta));       // load alpha_r and duplicate
				ymm2 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load alpha_i and duplicate
				//Multiply ymm4 with beta
				xmm0 = _mm_loadu_pd((double *)(tC)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm4 = _mm256_add_pd(ymm4, ymm0);
				//Multiply ymm8 with beta
				xmm0 = _mm_loadu_pd((double *)(tC + 1)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + 1 + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm8 = _mm256_add_pd(ymm8, ymm0);

				//Multiply ymm12 with beta
				xmm0 = _mm_loadu_pd((double *)(tC + 2)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + 2 + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm12 = _mm256_add_pd(ymm12, ymm0);

				//transpose left 3x2
				_mm_storeu_pd((double *)(tC  ), _mm256_castpd256_pd128(ymm4));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm8));
				_mm_storeu_pd((double *)(tC+2), _mm256_castpd256_pd128(ymm12));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm4,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm8,1));
				_mm_storeu_pd((double *)(tC+2), _mm256_extractf128_pd(ymm12, 1));
				tC += tc_inc_col;

				//Multiply ymm5 with beta
				xmm0 = _mm_loadu_pd((double *)(tC)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm5 = _mm256_add_pd(ymm5, ymm0);
				//Multiply ymm9 with beta
				xmm0 = _mm_loadu_pd((double *)(tC + 1)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + 1 + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm9 = _mm256_add_pd(ymm9, ymm0);

				//Multiply ymm13 with beta
				xmm0 = _mm_loadu_pd((double *)(tC + 2)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + 2 + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm13 = _mm256_add_pd(ymm13, ymm0);

				//transpose right 3x2
				_mm_storeu_pd((double *)(tC  ), _mm256_castpd256_pd128(ymm5));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm9));
				_mm_storeu_pd((double *)(tC+2), _mm256_castpd256_pd128(ymm13));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm5,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm9,1));
				_mm_storeu_pd((double *)(tC+2), _mm256_extractf128_pd(ymm13, 1));
			}

		}
		else
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				_mm256_storeu_pd((double *)(tC), ymm4);
				_mm256_storeu_pd((double *)(tC + 2), ymm5);
				_mm256_storeu_pd((double *)(tC + tc_inc_row) ,  ymm8);
				_mm256_storeu_pd((double *)(tC + tc_inc_row + 2), ymm9);
				_mm256_storeu_pd((double *)(tC + tc_inc_row *2), ymm12);
				_mm256_storeu_pd((double *)(tC + tc_inc_row *2+ 2), ymm13);
			}
			else{
				/* (br + bi) C + (ar + ai) AB */
				ymm0 = _mm256_broadcast_sd((double const *)(beta));       // load beta_r and duplicate
				ymm1 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load beta_i and duplicate

				ymm2 = _mm256_loadu_pd((double const *)(tC));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 =_mm256_mul_pd(ymm1, ymm3);
				ymm4 = _mm256_add_pd(ymm4, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+2));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm5 = _mm256_add_pd(ymm5, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm8 = _mm256_add_pd(ymm8, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row + 2));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm9 = _mm256_add_pd(ymm9, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row*2));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm12 = _mm256_add_pd(ymm12, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row*2 +2));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm13 = _mm256_add_pd(ymm13, _mm256_addsub_pd(ymm2, ymm3));

				_mm256_storeu_pd((double *)(tC), ymm4);
				_mm256_storeu_pd((double *)(tC + 2), ymm5);
				_mm256_storeu_pd((double *)(tC + tc_inc_row) ,  ymm8);
				_mm256_storeu_pd((double *)(tC + tc_inc_row + 2), ymm9);
				_mm256_storeu_pd((double *)(tC + tc_inc_row *2), ymm12);
				_mm256_storeu_pd((double *)(tC + tc_inc_row *2+ 2), ymm13);
			}
		}
	}

	consider_edge_cases:
	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		dcomplex* restrict cij = c + j_edge*cs_c;
		dcomplex* restrict ai  = a;
		dcomplex* restrict bj  = b + n_iter * 4;

		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_zgemmsup_rv_zen_asm_3x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			bli_zgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
		}
	}

}

void bli_zgemmsup_rv_zen_asm_2x4n
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

	uint64_t k_iter = 0;


	uint64_t n_iter = n0 / 4;
	uint64_t n_left = n0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------
	//scratch registers
	__m256d ymm0, ymm1, ymm2, ymm3;
	__m256d ymm4, ymm5, ymm6, ymm7;
	__m256d ymm8, ymm9, ymm10, ymm11;
	__m128d xmm0, xmm3;

	dcomplex *tA = a;
	double *tAimag = &a->imag;
	dcomplex *tB = b;
	dcomplex *tC = c;
	for (n_iter = 0; n_iter < n0 / 4; n_iter++)
	{
		// clear scratch registers.
		ymm4 = _mm256_setzero_pd();
		ymm5 = _mm256_setzero_pd();
		ymm6 = _mm256_setzero_pd();
		ymm7 = _mm256_setzero_pd();
		ymm8 = _mm256_setzero_pd();
		ymm9 = _mm256_setzero_pd();
		ymm10 = _mm256_setzero_pd();
		ymm11 = _mm256_setzero_pd();

		dim_t ta_inc_row = rs_a;
		dim_t tb_inc_row = rs_b;
		dim_t tc_inc_row = rs_c;

		dim_t ta_inc_col = cs_a;
		dim_t tb_inc_col = cs_b;
		dim_t tc_inc_col = cs_c;

		tA = a;
		tAimag = &a->imag;
		tB = b + n_iter*tb_inc_col*4;
		tC = c + n_iter*tc_inc_col*4;
		for (k_iter = 0; k_iter <k0; k_iter++)
		{
			// The inner loop broadcasts the B matrix data and
			// multiplies it with the A matrix.
			// This loop is processing MR x K
			ymm0 = _mm256_loadu_pd((double const *)(tB + tb_inc_row * k_iter));
			ymm1 = _mm256_loadu_pd((double const *)(tB + tb_inc_row * k_iter +  2));

			//broadcasted matrix B elements are multiplied
			//with matrix A columns.
			ymm2 = _mm256_broadcast_sd((double const *)(tA));
			ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
			ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);

			ymm2 = _mm256_broadcast_sd((double const *)(tA + ta_inc_row));
			ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);
			ymm9 = _mm256_fmadd_pd(ymm1, ymm2, ymm9);

			//Compute imag values
			ymm2 = _mm256_broadcast_sd((double const *)(tAimag ));
			ymm6 = _mm256_fmadd_pd(ymm0, ymm2, ymm6);
			ymm7 = _mm256_fmadd_pd(ymm1, ymm2, ymm7);

			ymm2 = _mm256_broadcast_sd((double const *)(tAimag + ta_inc_row *2));
			ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);
			ymm11 = _mm256_fmadd_pd(ymm1, ymm2, ymm11);

			tA += ta_inc_col;
			tAimag += ta_inc_col*2;
		}
		ymm6 =_mm256_permute_pd(ymm6,  5);
		ymm7 =_mm256_permute_pd(ymm7,  5);
		ymm10 = _mm256_permute_pd(ymm10, 5);
		ymm11 = _mm256_permute_pd(ymm11, 5);

		// subtract/add even/odd elements
		ymm4 = _mm256_addsub_pd(ymm4, ymm6);
		ymm5 = _mm256_addsub_pd(ymm5, ymm7);

		ymm8 = _mm256_addsub_pd(ymm8, ymm10);
		ymm9 = _mm256_addsub_pd(ymm9, ymm11);

		// alpha, beta multiplication.

		/* (ar + ai) x AB */
		ymm0 = _mm256_broadcast_sd((double const *)(alpha));       // load alpha_r and duplicate
		ymm1 = _mm256_broadcast_sd((double const *)(&alpha->imag));    // load alpha_i and duplicate

		ymm3 = _mm256_permute_pd(ymm4, 5);
		ymm4 = _mm256_mul_pd(ymm0, ymm4);
		ymm3 =_mm256_mul_pd(ymm1, ymm3);
		ymm4 = _mm256_addsub_pd(ymm4, ymm3);

		ymm3 = _mm256_permute_pd(ymm5, 5);
		ymm5 = _mm256_mul_pd(ymm0, ymm5);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm5 = _mm256_addsub_pd(ymm5, ymm3);

		ymm3 = _mm256_permute_pd(ymm8, 5);
		ymm8 = _mm256_mul_pd(ymm0, ymm8);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm8 = _mm256_addsub_pd(ymm8, ymm3);

		ymm3 = _mm256_permute_pd(ymm9, 5);
		ymm9 = _mm256_mul_pd(ymm0, ymm9);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm9 = _mm256_addsub_pd(ymm9, ymm3);

		if(tc_inc_row == 1) //col stored
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				//transpose left 2x2
				_mm_storeu_pd((double *)(tC  ), _mm256_castpd256_pd128(ymm4));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm8));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm4,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm8,1));
				tC += tc_inc_col;

				//transpose right 2x2
				_mm_storeu_pd((double *)(tC  ), _mm256_castpd256_pd128(ymm5));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm9));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm5,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm9,1));
			}
			else{
				ymm1 = _mm256_broadcast_sd((double const *)(beta));       // load alpha_r and duplicate
				ymm2 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load alpha_i and duplicate
				//Multiply ymm4 with beta
				xmm0 = _mm_loadu_pd((double *)(tC)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm4 = _mm256_add_pd(ymm4, ymm0);
				//Multiply ymm8 with beta
				xmm0 = _mm_loadu_pd((double *)(tC + 1)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + 1 + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm8 = _mm256_add_pd(ymm8, ymm0);

				//transpose left 2x2
				_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm4));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm8));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC ) ,_mm256_extractf128_pd (ymm4,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm8,1));
				tC += tc_inc_col;


				//Multiply ymm5 with beta
				xmm0 = _mm_loadu_pd((double *)(tC)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm5 = _mm256_add_pd(ymm5, ymm0);
				//Multiply ymm9 with beta
				xmm0 = _mm_loadu_pd((double *)(tC + 1)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + 1 + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm9 = _mm256_add_pd(ymm9, ymm0);

				//transpose right 2x2
				_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm5));
				_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm9));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC ) ,_mm256_extractf128_pd (ymm5,1));
				_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm9,1));
			}

		}
		else
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				_mm256_storeu_pd((double *)(tC), ymm4);
				_mm256_storeu_pd((double *)(tC + 2), ymm5);
				_mm256_storeu_pd((double *)(tC + tc_inc_row) ,  ymm8);
				_mm256_storeu_pd((double *)(tC + tc_inc_row + 2), ymm9);
			}
			else{
				/* (br + bi) C + (ar + ai) AB */
				ymm0 = _mm256_broadcast_sd((double const *)(beta));       // load beta_r and duplicate
				ymm1 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load beta_i and duplicate

				ymm2 = _mm256_loadu_pd((double const *)(tC));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 =_mm256_mul_pd(ymm1, ymm3);
				ymm4 = _mm256_add_pd(ymm4, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+2));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm5 = _mm256_add_pd(ymm5, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm8 = _mm256_add_pd(ymm8, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row + 2));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm9 = _mm256_add_pd(ymm9, _mm256_addsub_pd(ymm2, ymm3));

				_mm256_storeu_pd((double *)(tC), ymm4);
				_mm256_storeu_pd((double *)(tC + 2), ymm5);
				_mm256_storeu_pd((double *)(tC + tc_inc_row) ,  ymm8);
				_mm256_storeu_pd((double *)(tC + tc_inc_row + 2), ymm9);
			}
		}
	}

	consider_edge_cases:
	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		dcomplex* restrict cij = c + j_edge*cs_c;
		dcomplex* restrict ai  = a;
		dcomplex* restrict bj  = b + n_iter * 4;

		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_zgemmsup_rv_zen_asm_2x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			bli_zgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
		}
	}

}

void bli_zgemmsup_rv_zen_asm_1x4n
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

	uint64_t k_iter = 0;

	uint64_t n_iter = n0 / 4;
	uint64_t n_left = n0 % 4;

	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------
	//scratch registers
	__m256d ymm0, ymm1, ymm2, ymm3;
	__m256d ymm4, ymm5, ymm6, ymm7;
	__m128d xmm0, xmm3;

	dcomplex *tA = a;
	double *tAimag = &a->imag;
	dcomplex *tB = b;
	dcomplex *tC = c;
	for (n_iter = 0; n_iter < n0 / 4; n_iter++)
	{
		// clear scratch registers.
		ymm4 = _mm256_setzero_pd();
		ymm5 = _mm256_setzero_pd();
		ymm6 = _mm256_setzero_pd();
		ymm7 = _mm256_setzero_pd();

		dim_t tb_inc_row = rs_b;
		dim_t tc_inc_row = rs_c;

		dim_t ta_inc_col = cs_a;
		dim_t tb_inc_col = cs_b;
		dim_t tc_inc_col = cs_c;

		tA = a;
		tAimag = &a->imag;
		tB = b + n_iter*tb_inc_col*4;
		tC = c + n_iter*tc_inc_col*4;
		for (k_iter = 0; k_iter <k0; k_iter++)
		{
			// The inner loop broadcasts the B matrix data and
			// multiplies it with the A matrix.
			// This loop is processing MR x K
			ymm0 = _mm256_loadu_pd((double const *)(tB + tb_inc_row * k_iter));
			ymm1 = _mm256_loadu_pd((double const *)(tB + tb_inc_row * k_iter +  2));

			//broadcasted matrix B elements are multiplied
			//with matrix A columns.
			ymm2 = _mm256_broadcast_sd((double const *)(tA));
			ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);
			ymm5 = _mm256_fmadd_pd(ymm1, ymm2, ymm5);

			//Compute imag values
			ymm2 = _mm256_broadcast_sd((double const *)(tAimag ));
			ymm6 = _mm256_fmadd_pd(ymm0, ymm2, ymm6);
			ymm7 = _mm256_fmadd_pd(ymm1, ymm2, ymm7);

			tA += ta_inc_col;
			tAimag += ta_inc_col*2;
		}
		ymm6 =_mm256_permute_pd(ymm6,  5);
		ymm7 =_mm256_permute_pd(ymm7,  5);

		// subtract/add even/odd elements
		ymm4 = _mm256_addsub_pd(ymm4, ymm6);
		ymm5 = _mm256_addsub_pd(ymm5, ymm7);

		// alpha, beta multiplication.

		/* (ar + ai) x AB */
		ymm0 = _mm256_broadcast_sd((double const *)(alpha));       // load alpha_r and duplicate
		ymm1 = _mm256_broadcast_sd((double const *)(&alpha->imag));    // load alpha_i and duplicate

		ymm3 = _mm256_permute_pd(ymm4, 5);
		ymm4 = _mm256_mul_pd(ymm0, ymm4);
		ymm3 =_mm256_mul_pd(ymm1, ymm3);
		ymm4 = _mm256_addsub_pd(ymm4, ymm3);

		ymm3 = _mm256_permute_pd(ymm5, 5);
		ymm5 = _mm256_mul_pd(ymm0, ymm5);
		ymm3 = _mm256_mul_pd(ymm1, ymm3);
		ymm5 = _mm256_addsub_pd(ymm5, ymm3);

		if(tc_inc_row == 1) //col stored
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				//transpose left 1x2
				_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm4));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC)  ,_mm256_extractf128_pd (ymm4,1));
				tC += tc_inc_col;

				//transpose right 1x2
				_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm5));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC)  ,_mm256_extractf128_pd (ymm5,1));
			}
			else{
				ymm1 = _mm256_broadcast_sd((double const *)(beta));       // load alpha_r and duplicate
				ymm2 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load alpha_i and duplicate
				//Multiply ymm4 with beta
				xmm0 = _mm_loadu_pd((double *)(tC)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm4 = _mm256_add_pd(ymm4, ymm0);

				_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm4));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC ) ,_mm256_extractf128_pd (ymm4,1));
				tC += tc_inc_col;

				//Multiply ymm5 with beta
				xmm0 = _mm_loadu_pd((double *)(tC)) ;
				xmm3 = _mm_loadu_pd((double *)(tC + tc_inc_col)) ;
				ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
				ymm3 = _mm256_permute_pd(ymm0, 5);
				ymm0 = _mm256_mul_pd(ymm1, ymm0);
				ymm3 = _mm256_mul_pd(ymm2, ymm3);
				ymm0 = _mm256_addsub_pd(ymm0, ymm3);
				ymm5 = _mm256_add_pd(ymm5, ymm0);

				_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm5));
				tC += tc_inc_col;

				_mm_storeu_pd((double *)(tC)  ,_mm256_extractf128_pd (ymm5,1));
			}

		}
		else
		{
			if(beta->real == 0.0 && beta->imag == 0.0)
			{
				_mm256_storeu_pd((double *)(tC), ymm4);
				_mm256_storeu_pd((double *)(tC + 2), ymm5);
			}
			else{
				/* (br + bi) C + (ar + ai) AB */
				ymm0 = _mm256_broadcast_sd((double const *)(beta));       // load beta_r and duplicate
				ymm1 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load beta_i and duplicate

				ymm2 = _mm256_loadu_pd((double const *)(tC));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 =_mm256_mul_pd(ymm1, ymm3);
				ymm4 = _mm256_add_pd(ymm4, _mm256_addsub_pd(ymm2, ymm3));

				ymm2 = _mm256_loadu_pd((double const *)(tC+2));
				ymm3 = _mm256_permute_pd(ymm2, 5);
				ymm2 = _mm256_mul_pd(ymm0, ymm2);
				ymm3 = _mm256_mul_pd(ymm1, ymm3);
				ymm5 = _mm256_add_pd(ymm5, _mm256_addsub_pd(ymm2, ymm3));

				_mm256_storeu_pd((double *)(tC), ymm4);
				_mm256_storeu_pd((double *)(tC + 2), ymm5);
			}
		}
	}

	consider_edge_cases:
	// Handle edge cases in the m dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		dcomplex* restrict cij = c + j_edge*cs_c;
		dcomplex* restrict ai  = a;
		dcomplex* restrict bj  = b + n_iter * 4;

		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;
			bli_zgemmsup_rv_zen_asm_1x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			bli_zgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, m0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
		}
	}
}

void bli_zgemmsup_rv_zen_asm_3x2
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
	uint64_t k_iter = 0;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;



	// -------------------------------------------------------------------------
	//scratch registers
	__m256d ymm0, ymm1, ymm2, ymm3;
	__m256d ymm4, ymm6;
	__m256d ymm8, ymm10;
	__m256d ymm12, ymm14;
	__m128d xmm0, xmm3;

	dcomplex *tA = a;
	double *tAimag = &a->imag;
	dcomplex *tB = b;
	dcomplex *tC = c;
	// clear scratch registers.
	ymm4 = _mm256_setzero_pd();
	ymm6 = _mm256_setzero_pd();
	ymm8 = _mm256_setzero_pd();
	ymm10 = _mm256_setzero_pd();
	ymm12 = _mm256_setzero_pd();
	ymm14 = _mm256_setzero_pd();

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
		ymm0 = _mm256_loadu_pd((double const *)(tB + tb_inc_row * k_iter));

		//broadcasted matrix B elements are multiplied
		//with matrix A columns.
		ymm2 = _mm256_broadcast_sd((double const *)(tA));
		ymm4 = _mm256_fmadd_pd(ymm0, ymm2, ymm4);

		ymm2 = _mm256_broadcast_sd((double const *)(tA + ta_inc_row));
		ymm8 = _mm256_fmadd_pd(ymm0, ymm2, ymm8);

		ymm2 = _mm256_broadcast_sd((double const *)(tA + ta_inc_row*2));
		ymm12 = _mm256_fmadd_pd(ymm0, ymm2, ymm12);

		//Compute imag values
		ymm2 = _mm256_broadcast_sd((double const *)(tAimag ));
		ymm6 = _mm256_fmadd_pd(ymm0, ymm2, ymm6);

		ymm2 = _mm256_broadcast_sd((double const *)(tAimag + ta_inc_row *2));
		ymm10 = _mm256_fmadd_pd(ymm0, ymm2, ymm10);

		ymm2 = _mm256_broadcast_sd((double const *)(tAimag + ta_inc_row *4));
		ymm14 = _mm256_fmadd_pd(ymm0, ymm2, ymm14);
		tA += ta_inc_col;
		tAimag += ta_inc_col*2;
	}
	ymm6 =_mm256_permute_pd(ymm6,  5);
	ymm10 = _mm256_permute_pd(ymm10, 5);
	ymm14 = _mm256_permute_pd(ymm14, 5);

	// subtract/add even/odd elements
	ymm4 = _mm256_addsub_pd(ymm4, ymm6);

	ymm8 = _mm256_addsub_pd(ymm8, ymm10);

	ymm12 = _mm256_addsub_pd( ymm12, ymm14);

	// alpha, beta multiplication.

	/* (ar + ai) x AB */
	ymm0 = _mm256_broadcast_sd((double const *)(alpha));       // load alpha_r and duplicate
	ymm1 = _mm256_broadcast_sd((double const *)(&alpha->imag));    // load alpha_i and duplicate

	ymm3 = _mm256_permute_pd(ymm4, 5);
	ymm4 = _mm256_mul_pd(ymm0, ymm4);
	ymm3 =_mm256_mul_pd(ymm1, ymm3);
	ymm4 = _mm256_addsub_pd(ymm4, ymm3);

	ymm3 = _mm256_permute_pd(ymm8, 5);
	ymm8 = _mm256_mul_pd(ymm0, ymm8);
	ymm3 = _mm256_mul_pd(ymm1, ymm3);
	ymm8 = _mm256_addsub_pd(ymm8, ymm3);

	ymm3 = _mm256_permute_pd(ymm12, 5);
	ymm12 = _mm256_mul_pd(ymm0, ymm12);
	ymm3 = _mm256_mul_pd(ymm1, ymm3);
	ymm12 = _mm256_addsub_pd(ymm12, ymm3);

	if(tc_inc_row == 1) //col stored
	{
		if(beta->real == 0.0 && beta->imag == 0.0)
		{
			//transpose left 3x2
			_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm4));
			_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm8));
			_mm_storeu_pd((double *)(tC+2), _mm256_castpd256_pd128(ymm12));
			tC += tc_inc_col;

			_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm4,1));
			_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm8,1));
			_mm_storeu_pd((double *)(tC+2), _mm256_extractf128_pd(ymm12, 1));
		}
		else{
			ymm1 = _mm256_broadcast_sd((double const *)(beta));       // load alpha_r and duplicate
			ymm2 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load alpha_i and duplicate
			//Multiply ymm4 with beta
			xmm0 = _mm_loadu_pd((double *)(tC)) ;
			xmm3 = _mm_loadu_pd((double *)(tC + tc_inc_col)) ;
			ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
			ymm3 = _mm256_permute_pd(ymm0, 5);
			ymm0 = _mm256_mul_pd(ymm1, ymm0);
			ymm3 = _mm256_mul_pd(ymm2, ymm3);
			ymm0 = _mm256_addsub_pd(ymm0, ymm3);
			ymm4 = _mm256_add_pd(ymm4, ymm0);
			//Multiply ymm8 with beta
			xmm0 = _mm_loadu_pd((double *)(tC + 1)) ;
			xmm3 = _mm_loadu_pd((double *)(tC + 1 + tc_inc_col)) ;
			ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
			ymm3 = _mm256_permute_pd(ymm0, 5);
			ymm0 = _mm256_mul_pd(ymm1, ymm0);
			ymm3 = _mm256_mul_pd(ymm2, ymm3);
			ymm0 = _mm256_addsub_pd(ymm0, ymm3);
			ymm8 = _mm256_add_pd(ymm8, ymm0);

			//Multiply ymm12 with beta
			xmm0 = _mm_loadu_pd((double *)(tC + 2)) ;
			xmm3 = _mm_loadu_pd((double *)(tC + 2 + tc_inc_col)) ;
			ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm3, 1) ;
			ymm3 = _mm256_permute_pd(ymm0, 5);
			ymm0 = _mm256_mul_pd(ymm1, ymm0);
			ymm3 = _mm256_mul_pd(ymm2, ymm3);
			ymm0 = _mm256_addsub_pd(ymm0, ymm3);
			ymm12 = _mm256_add_pd(ymm12, ymm0);

			_mm_storeu_pd((double *)(tC), _mm256_castpd256_pd128(ymm4));
			_mm_storeu_pd((double *)(tC+1), _mm256_castpd256_pd128(ymm8));
			_mm_storeu_pd((double *)(tC+2), _mm256_castpd256_pd128(ymm12));
			tC += tc_inc_col;
			_mm_storeu_pd((double *)(tC  ),_mm256_extractf128_pd (ymm4,1));
			_mm_storeu_pd((double *)(tC+1)  ,_mm256_extractf128_pd (ymm8,1));
			_mm_storeu_pd((double *)(tC+2), _mm256_extractf128_pd(ymm12, 1));
		}
	}
	else
	{
		if(beta->real == 0.0 && beta->imag == 0.0)
		{
			_mm256_storeu_pd((double *)(tC), ymm4);
			_mm256_storeu_pd((double *)(tC + tc_inc_row  ),  ymm8);
			_mm256_storeu_pd((double *)(tC + tc_inc_row *2), ymm12);
		}
		else{
			/* (br + bi) C + (ar + ai) AB */
			ymm0 = _mm256_broadcast_sd((double const *)(beta));       // load beta_r and duplicate
			ymm1 = _mm256_broadcast_sd((double const *)(&beta->imag));    // load beta_i and duplicate

			ymm2 = _mm256_loadu_pd((double const *)(tC));
			ymm3 = _mm256_permute_pd(ymm2, 5);
			ymm2 = _mm256_mul_pd(ymm0, ymm2);
			ymm3 =_mm256_mul_pd(ymm1, ymm3);
			ymm4 = _mm256_add_pd(ymm4, _mm256_addsub_pd(ymm2, ymm3));

			ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row));
			ymm3 = _mm256_permute_pd(ymm2, 5);
			ymm2 = _mm256_mul_pd(ymm0, ymm2);
			ymm3 = _mm256_mul_pd(ymm1, ymm3);
			ymm8 = _mm256_add_pd(ymm8, _mm256_addsub_pd(ymm2, ymm3));

			ymm2 = _mm256_loadu_pd((double const *)(tC+tc_inc_row*2));
			ymm3 = _mm256_permute_pd(ymm2, 5);
			ymm2 = _mm256_mul_pd(ymm0, ymm2);
			ymm3 = _mm256_mul_pd(ymm1, ymm3);
			ymm12 = _mm256_add_pd(ymm12, _mm256_addsub_pd(ymm2, ymm3));

			_mm256_storeu_pd((double *)(tC), ymm4);
			_mm256_storeu_pd((double *)(tC + tc_inc_row) ,  ymm8);
			_mm256_storeu_pd((double *)(tC + tc_inc_row *2), ymm12);
		}
	}
}
