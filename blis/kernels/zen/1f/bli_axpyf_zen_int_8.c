/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2016 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include "immintrin.h"
#include "blis.h"

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
	__m256d v;
	double  d[4] __attribute__((aligned(64)));
} v4df_t;

// -----------------------------------------------------------------------------

void bli_saxpyf_zen_int_8
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a00, inc_t inca, inc_t lda,
       const void*   x00, inc_t incx,
             void*   y00, inc_t incy,
       const cntx_t* cntx
     )
{
	const float* restrict alpha = (float *)alpha0;
	float* restrict a     = (float *)a00;
	float* restrict x     = (float *)x00;
	float* restrict y     = (float *)y00;

	const dim_t      fuse_fac       = 8;
	const dim_t      n_elem_per_reg = 8;
	const dim_t      n_iter_unroll  = 1;

	dim_t            i;
	dim_t            m_viter;
	dim_t            m_left;

	float*  restrict a0;
	float*  restrict a1;
	float*  restrict a2;
	float*  restrict a3;
	float*  restrict a4;
	float*  restrict a5;
	float*  restrict a6;
	float*  restrict a7;

	float*  restrict y0;

	v8sf_t           chi0v, chi1v, chi2v, chi3v;
	v8sf_t           chi4v, chi5v, chi6v, chi7v;

	v8sf_t           a0v, a1v, a2v, a3v;
	v8sf_t           a4v, a5v, a6v, a7v;
	v8sf_t           y0v;

	float            chi0, chi1, chi2, chi3;
	float            chi4, chi5, chi6, chi7;

	// If either dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim2( m, b_n ) || PASTEMAC(s,eq0)( *alpha ) ) return;

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over axpyv.
	if ( b_n != fuse_fac )
	{
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();
        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_FLOAT, BLIS_AXPYV_KER, cntx );

		for ( i = 0; i < b_n; ++i )
		{
			float* a1   = a + (0  )*inca + (i  )*lda;
			float* chi1 = x + (i  )*incx;
			float* y1   = y + (0  )*incy;
			float  alpha_chi1;

			PASTEMAC(s,copycjs)( conjx, *chi1, alpha_chi1 );
			PASTEMAC(s,scals)( *alpha, alpha_chi1 );

			f
			(
			  conja,
			  m,
			  &alpha_chi1,
			  a1, inca,
			  y1, incy,
			  cntx
			);
		}

		return;
	}

	// At this point, we know that b_n is exactly equal to the fusing factor.

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	m_viter = ( m ) / ( n_elem_per_reg * n_iter_unroll );
	m_left  = ( m ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override m_viter and m_left to use scalar code
	// for all iterations.
	if ( inca != 1 || incy != 1 )
	{
		m_viter = 0;
		m_left  = m;
	}

	a0   = a + 0*lda;
	a1   = a + 1*lda;
	a2   = a + 2*lda;
	a3   = a + 3*lda;
	a4   = a + 4*lda;
	a5   = a + 5*lda;
	a6   = a + 6*lda;
	a7   = a + 7*lda;
	y0   = y;

	chi0 = *( (float *)x + 0*incx );
	chi1 = *( (float *)x + 1*incx );
	chi2 = *( (float *)x + 2*incx );
	chi3 = *( (float *)x + 3*incx );
	chi4 = *( (float *)x + 4*incx );
	chi5 = *( (float *)x + 5*incx );
	chi6 = *( (float *)x + 6*incx );
	chi7 = *( (float *)x + 7*incx );

	// Scale each chi scalar by alpha.
	PASTEMAC(s,scals)( *alpha, chi0 );
	PASTEMAC(s,scals)( *alpha, chi1 );
	PASTEMAC(s,scals)( *alpha, chi2 );
	PASTEMAC(s,scals)( *alpha, chi3 );
	PASTEMAC(s,scals)( *alpha, chi4 );
	PASTEMAC(s,scals)( *alpha, chi5 );
	PASTEMAC(s,scals)( *alpha, chi6 );
	PASTEMAC(s,scals)( *alpha, chi7 );

	// Broadcast the (alpha*chi?) scalars to all elements of vector registers.
	chi0v.v = _mm256_broadcast_ss( &chi0 );
	chi1v.v = _mm256_broadcast_ss( &chi1 );
	chi2v.v = _mm256_broadcast_ss( &chi2 );
	chi3v.v = _mm256_broadcast_ss( &chi3 );
	chi4v.v = _mm256_broadcast_ss( &chi4 );
	chi5v.v = _mm256_broadcast_ss( &chi5 );
	chi6v.v = _mm256_broadcast_ss( &chi6 );
	chi7v.v = _mm256_broadcast_ss( &chi7 );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	for ( i = 0; i < m_viter; ++i )
	{
		// Load the input values.
		y0v.v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
		a0v.v = _mm256_loadu_ps( a0 + 0*n_elem_per_reg );
		a1v.v = _mm256_loadu_ps( a1 + 0*n_elem_per_reg );
		a2v.v = _mm256_loadu_ps( a2 + 0*n_elem_per_reg );
		a3v.v = _mm256_loadu_ps( a3 + 0*n_elem_per_reg );
		a4v.v = _mm256_loadu_ps( a4 + 0*n_elem_per_reg );
		a5v.v = _mm256_loadu_ps( a5 + 0*n_elem_per_reg );
		a6v.v = _mm256_loadu_ps( a6 + 0*n_elem_per_reg );
		a7v.v = _mm256_loadu_ps( a7 + 0*n_elem_per_reg );

		// perform : y += alpha * x;
		y0v.v = _mm256_fmadd_ps( a0v.v, chi0v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a1v.v, chi1v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a2v.v, chi2v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a3v.v, chi3v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a4v.v, chi4v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a5v.v, chi5v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a6v.v, chi6v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a7v.v, chi7v.v, y0v.v );

		// Store the output.
		_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), y0v.v );

		y0 += n_elem_per_reg;
		a0 += n_elem_per_reg;
		a1 += n_elem_per_reg;
		a2 += n_elem_per_reg;
		a3 += n_elem_per_reg;
		a4 += n_elem_per_reg;
		a5 += n_elem_per_reg;
		a6 += n_elem_per_reg;
		a7 += n_elem_per_reg;
	}

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < m_left ; ++i )
	{
		float       y0c = *y0;

		const float a0c = *a0;
		const float a1c = *a1;
		const float a2c = *a2;
		const float a3c = *a3;
		const float a4c = *a4;
		const float a5c = *a5;
		const float a6c = *a6;
		const float a7c = *a7;

		y0c += chi0 * a0c;
		y0c += chi1 * a1c;
		y0c += chi2 * a2c;
		y0c += chi3 * a3c;
		y0c += chi4 * a4c;
		y0c += chi5 * a5c;
		y0c += chi6 * a6c;
		y0c += chi7 * a7c;

		*y0 = y0c;

		a0 += inca;
		a1 += inca;
		a2 += inca;
		a3 += inca;
		a4 += inca;
		a5 += inca;
		a6 += inca;
		a7 += inca;
		y0 += incy;
	}
}

// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_8
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a00, inc_t inca, inc_t lda,
       const void*   x00, inc_t incx,
             void*   y00, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* restrict alpha = (double *) alpha0;
	double* restrict a     = (double *) a00;
	double* restrict x     = (double *) x00;
	double* restrict y     = (double *) y00;

	const dim_t      fuse_fac       = 8;
	const dim_t      n_elem_per_reg = 4;
	const dim_t      n_iter_unroll[4]  = {4, 3, 2, 1};

	dim_t            i;
	dim_t            m_viter[4];
	dim_t            m_left = m;

	double* restrict av[8] __attribute__((aligned(64)));

	double* restrict y0;

	v4df_t           chiv[8], a_vec[32], yv[4];

	double           chi[8] __attribute__((aligned(64)));

	// If either dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim2( m, b_n ) || PASTEMAC(d,eq0)( *alpha ) ) return;

	/*
	  If b_n is not equal to the fusing factor, then perform the entire
	  operation as axpyv or perform the operation using axpyf kernels with
	  lower fuse factor.
	*/
	if ( b_n < fuse_fac )
	{
		if (b_n >= 5)
		{
			dim_t fuse_fac = 5;

			bli_daxpyf_zen_int_5
			(
			  conja,
			  conjx,
			  m,
			  fuse_fac,
			  alpha,
			  a, inca, lda,
			  x, incx,
			  y, incy,
			  cntx
			);

			a = a + (fuse_fac * lda);
			x = x + (fuse_fac * incx);

			b_n -= fuse_fac;
		}

		if (b_n == 4)
		{
			dim_t fuse_fac = 4;

			bli_daxpyf_zen_int_16x4
			(
			  conja,
			  conjx,
			  m,
			  fuse_fac,
			  alpha,
			  a, inca, lda,
			  x, incx,
			  y, incy,
			  cntx
			);

			a = a + (fuse_fac * lda);
			x = x + (fuse_fac * incx);

			b_n -= fuse_fac;
		}

		if (b_n >= 2)
		{
			dim_t fuse_fac = 2;

			bli_daxpyf_zen_int_16x2
			(
			  conja,
			  conjx,
			  m, fuse_fac,
			  alpha, a, inca, lda,
			  x, incx,
			  y, incy,
			  cntx
			);

			a = a + (fuse_fac * lda);
			x = x + (fuse_fac * incx);

			b_n -= fuse_fac;

		}

		if (b_n == 1)
		{
			// Query the context if it is NULL. This will be necessary for Zen architectures
        	if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();
        	axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

			double* a1   = a;
			double* chi1 = x;
			double* y1   = y;
			double  alpha_chi1;

			PASTEMAC(d,copycjs)( conjx, *chi1, alpha_chi1 );
			PASTEMAC(d,scals)( *alpha, alpha_chi1 );

			f
			(
			  conja,
			  m,
			  &alpha_chi1,
			  a1, inca,
			  y1, incy,
			  cntx
			);
		}

		return;
	}
	else if ( b_n > fuse_fac )
	{
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();
        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

		for ( i = 0; i < b_n; ++i )
		{
			double* a1   = a + (0  )*inca + (i  )*lda;
			double* chi1 = x + (i  )*incx;
			double* y1   = y + (0  )*incy;
			double  alpha_chi1;

			bli_dcopycjs( conjx, *chi1, alpha_chi1 );
			bli_dscals( *alpha, alpha_chi1 );

			f
			(
			  conja,
			  m,
			  &alpha_chi1,
			  a1, inca,
			  y1, incy,
			  cntx
			);
		}

		return;
	}

	// At this point, we know that b_n is exactly equal to the fusing factor.

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	m_viter[0] = ( m_left ) / ( n_elem_per_reg * n_iter_unroll[0] );
	m_left  = ( m_left ) % ( n_elem_per_reg * n_iter_unroll[0] );

	m_viter[1] = ( m_left ) / ( n_elem_per_reg * n_iter_unroll[1] );
	m_left  = ( m_left ) % ( n_elem_per_reg * n_iter_unroll[1] );

	m_viter[2] = ( m_left ) / ( n_elem_per_reg * n_iter_unroll[2] );
	m_left  = ( m_left ) % ( n_elem_per_reg * n_iter_unroll[2] );

	m_viter[3] = ( m_left ) / ( n_elem_per_reg * n_iter_unroll[3] );
	m_left  = ( m_left ) % ( n_elem_per_reg * n_iter_unroll[3] );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override m_viter and m_left to use scalar code
	// for all iterations.
	if ( inca != 1 || incy != 1 )
	{
		m_viter[0] = m_viter[1] = m_viter[2] = m_viter[3] = 0;
		m_left  = m;
	}

	// av points to the 8 columns under consideration
	av[0]   = a + 0*lda;
	av[1]   = a + 1*lda;
	av[2]   = a + 2*lda;
	av[3]   = a + 3*lda;
	av[4]   = a + 4*lda;
	av[5]   = a + 5*lda;
	av[6]   = a + 6*lda;
	av[7]   = a + 7*lda;
	y0   = y;

	chi[0] = *( x + 0*incx );
	chi[1] = *( x + 1*incx );
	chi[2] = *( x + 2*incx );
	chi[3] = *( x + 3*incx );
	chi[4] = *( x + 4*incx );
	chi[5] = *( x + 5*incx );
	chi[6] = *( x + 6*incx );
	chi[7] = *( x + 7*incx );

	// Scale each chi scalar by alpha.
	PASTEMAC(d,scals)( *alpha, chi[0] );
	PASTEMAC(d,scals)( *alpha, chi[1] );
	PASTEMAC(d,scals)( *alpha, chi[2] );
	PASTEMAC(d,scals)( *alpha, chi[3] );
	PASTEMAC(d,scals)( *alpha, chi[4] );
	PASTEMAC(d,scals)( *alpha, chi[5] );
	PASTEMAC(d,scals)( *alpha, chi[6] );
	PASTEMAC(d,scals)( *alpha, chi[7] );

	// Broadcast the (alpha*chi?) scalars to all elements of vector registers.
	chiv[0].v = _mm256_broadcast_sd( &chi[0] );
	chiv[1].v = _mm256_broadcast_sd( &chi[1] );
	chiv[2].v = _mm256_broadcast_sd( &chi[2] );
	chiv[3].v = _mm256_broadcast_sd( &chi[3] );
	chiv[4].v = _mm256_broadcast_sd( &chi[4] );
	chiv[5].v = _mm256_broadcast_sd( &chi[5] );
	chiv[6].v = _mm256_broadcast_sd( &chi[6] );
	chiv[7].v = _mm256_broadcast_sd( &chi[7] );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	// 16 elements of the result are computed per iteration
	for ( i = 0; i < m_viter[0]; ++i )
	{
		// Load the input values.
		yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
		yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
		yv[2].v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
		yv[3].v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

		a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
		a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
		a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
		a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
		a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );
		a_vec[5].v = _mm256_loadu_pd( av[5] + 0*n_elem_per_reg );
		a_vec[6].v = _mm256_loadu_pd( av[6] + 0*n_elem_per_reg );
		a_vec[7].v = _mm256_loadu_pd( av[7] + 0*n_elem_per_reg );
		
		a_vec[8].v = _mm256_loadu_pd( av[0] + 1*n_elem_per_reg );
		a_vec[9].v = _mm256_loadu_pd( av[1] + 1*n_elem_per_reg );
		a_vec[10].v = _mm256_loadu_pd( av[2] + 1*n_elem_per_reg );
		a_vec[11].v = _mm256_loadu_pd( av[3] + 1*n_elem_per_reg );
		a_vec[12].v = _mm256_loadu_pd( av[4] + 1*n_elem_per_reg );
		a_vec[13].v = _mm256_loadu_pd( av[5] + 1*n_elem_per_reg );
		a_vec[14].v = _mm256_loadu_pd( av[6] + 1*n_elem_per_reg );
		a_vec[15].v = _mm256_loadu_pd( av[7] + 1*n_elem_per_reg );

		a_vec[16].v = _mm256_loadu_pd( av[0] + 2*n_elem_per_reg );
		a_vec[17].v = _mm256_loadu_pd( av[1] + 2*n_elem_per_reg );
		a_vec[18].v = _mm256_loadu_pd( av[2] + 2*n_elem_per_reg );
		a_vec[19].v = _mm256_loadu_pd( av[3] + 2*n_elem_per_reg );
		a_vec[20].v = _mm256_loadu_pd( av[4] + 2*n_elem_per_reg );
		a_vec[21].v = _mm256_loadu_pd( av[5] + 2*n_elem_per_reg );
		a_vec[22].v = _mm256_loadu_pd( av[6] + 2*n_elem_per_reg );
		a_vec[23].v = _mm256_loadu_pd( av[7] + 2*n_elem_per_reg );

		a_vec[24].v = _mm256_loadu_pd( av[0] + 3*n_elem_per_reg );
		a_vec[25].v = _mm256_loadu_pd( av[1] + 3*n_elem_per_reg );
		a_vec[26].v = _mm256_loadu_pd( av[2] + 3*n_elem_per_reg );
		a_vec[27].v = _mm256_loadu_pd( av[3] + 3*n_elem_per_reg );
		a_vec[28].v = _mm256_loadu_pd( av[4] + 3*n_elem_per_reg );
		a_vec[29].v = _mm256_loadu_pd( av[5] + 3*n_elem_per_reg );
		a_vec[30].v = _mm256_loadu_pd( av[6] + 3*n_elem_per_reg );
		a_vec[31].v = _mm256_loadu_pd( av[7] + 3*n_elem_per_reg );

		// perform : y += alpha * x;
		yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[5].v, chiv[5].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[6].v, chiv[6].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[7].v, chiv[7].v, yv[0].v );

		yv[1].v = _mm256_fmadd_pd( a_vec[8].v, chiv[0].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[9].v, chiv[1].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[10].v, chiv[2].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[11].v, chiv[3].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[12].v, chiv[4].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[13].v, chiv[5].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[14].v, chiv[6].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[15].v, chiv[7].v, yv[1].v );

		yv[2].v = _mm256_fmadd_pd( a_vec[16].v, chiv[0].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[17].v, chiv[1].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[18].v, chiv[2].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[19].v, chiv[3].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[20].v, chiv[4].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[21].v, chiv[5].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[22].v, chiv[6].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[23].v, chiv[7].v, yv[2].v );

		yv[3].v = _mm256_fmadd_pd( a_vec[24].v, chiv[0].v, yv[3].v );
		yv[3].v = _mm256_fmadd_pd( a_vec[25].v, chiv[1].v, yv[3].v );
		yv[3].v = _mm256_fmadd_pd( a_vec[26].v, chiv[2].v, yv[3].v );
		yv[3].v = _mm256_fmadd_pd( a_vec[27].v, chiv[3].v, yv[3].v );
		yv[3].v = _mm256_fmadd_pd( a_vec[28].v, chiv[4].v, yv[3].v );
		yv[3].v = _mm256_fmadd_pd( a_vec[29].v, chiv[5].v, yv[3].v );
		yv[3].v = _mm256_fmadd_pd( a_vec[30].v, chiv[6].v, yv[3].v );
		yv[3].v = _mm256_fmadd_pd( a_vec[31].v, chiv[7].v, yv[3].v );

		// Store the output.
		_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );
		_mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1].v );
		_mm256_storeu_pd( (y0 + 2*n_elem_per_reg), yv[2].v );
		_mm256_storeu_pd( (y0 + 3*n_elem_per_reg), yv[3].v );

		y0 += n_elem_per_reg * n_iter_unroll[0];
		av[0] += n_elem_per_reg * n_iter_unroll[0];
		av[1] += n_elem_per_reg * n_iter_unroll[0];
		av[2] += n_elem_per_reg * n_iter_unroll[0];
		av[3] += n_elem_per_reg * n_iter_unroll[0];
		av[4] += n_elem_per_reg * n_iter_unroll[0];
		av[5] += n_elem_per_reg * n_iter_unroll[0];
		av[6] += n_elem_per_reg * n_iter_unroll[0];
		av[7] += n_elem_per_reg * n_iter_unroll[0];
	}

	// 12 elements of the result are computed per iteration
	for ( i = 0; i < m_viter[1]; ++i )
	{
		// Load the input values.
		yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
		yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
		yv[2].v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );

		a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
		a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
		a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
		a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
		a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );
		a_vec[5].v = _mm256_loadu_pd( av[5] + 0*n_elem_per_reg );
		a_vec[6].v = _mm256_loadu_pd( av[6] + 0*n_elem_per_reg );
		a_vec[7].v = _mm256_loadu_pd( av[7] + 0*n_elem_per_reg );
		
		a_vec[8].v = _mm256_loadu_pd( av[0] + 1*n_elem_per_reg );
		a_vec[9].v = _mm256_loadu_pd( av[1] + 1*n_elem_per_reg );
		a_vec[10].v = _mm256_loadu_pd( av[2] + 1*n_elem_per_reg );
		a_vec[11].v = _mm256_loadu_pd( av[3] + 1*n_elem_per_reg );
		a_vec[12].v = _mm256_loadu_pd( av[4] + 1*n_elem_per_reg );
		a_vec[13].v = _mm256_loadu_pd( av[5] + 1*n_elem_per_reg );
		a_vec[14].v = _mm256_loadu_pd( av[6] + 1*n_elem_per_reg );
		a_vec[15].v = _mm256_loadu_pd( av[7] + 1*n_elem_per_reg );

		a_vec[16].v = _mm256_loadu_pd( av[0] + 2*n_elem_per_reg );
		a_vec[17].v = _mm256_loadu_pd( av[1] + 2*n_elem_per_reg );
		a_vec[18].v = _mm256_loadu_pd( av[2] + 2*n_elem_per_reg );
		a_vec[19].v = _mm256_loadu_pd( av[3] + 2*n_elem_per_reg );
		a_vec[20].v = _mm256_loadu_pd( av[4] + 2*n_elem_per_reg );
		a_vec[21].v = _mm256_loadu_pd( av[5] + 2*n_elem_per_reg );
		a_vec[22].v = _mm256_loadu_pd( av[6] + 2*n_elem_per_reg );
		a_vec[23].v = _mm256_loadu_pd( av[7] + 2*n_elem_per_reg );

		// perform : y += alpha * x;
		yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[5].v, chiv[5].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[6].v, chiv[6].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[7].v, chiv[7].v, yv[0].v );

		yv[1].v = _mm256_fmadd_pd( a_vec[8].v, chiv[0].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[9].v, chiv[1].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[10].v, chiv[2].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[11].v, chiv[3].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[12].v, chiv[4].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[13].v, chiv[5].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[14].v, chiv[6].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[15].v, chiv[7].v, yv[1].v );

		yv[2].v = _mm256_fmadd_pd( a_vec[16].v, chiv[0].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[17].v, chiv[1].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[18].v, chiv[2].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[19].v, chiv[3].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[20].v, chiv[4].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[21].v, chiv[5].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[22].v, chiv[6].v, yv[2].v );
		yv[2].v = _mm256_fmadd_pd( a_vec[23].v, chiv[7].v, yv[2].v );

		// Store the output.
		_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );
		_mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1].v );
		_mm256_storeu_pd( (y0 + 2*n_elem_per_reg), yv[2].v );

		y0 += n_elem_per_reg * n_iter_unroll[1];
		av[0] += n_elem_per_reg * n_iter_unroll[1];
		av[1] += n_elem_per_reg * n_iter_unroll[1];
		av[2] += n_elem_per_reg * n_iter_unroll[1];
		av[3] += n_elem_per_reg * n_iter_unroll[1];
		av[4] += n_elem_per_reg * n_iter_unroll[1];
		av[5] += n_elem_per_reg * n_iter_unroll[1];
		av[6] += n_elem_per_reg * n_iter_unroll[1];
		av[7] += n_elem_per_reg * n_iter_unroll[1];
	}

	// 8 elements of the result are computed per iteration
	for ( i = 0; i < m_viter[2]; ++i )
	{
		// Load the input values.
		yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
		yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

		a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
		a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
		a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
		a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
		a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );
		a_vec[5].v = _mm256_loadu_pd( av[5] + 0*n_elem_per_reg );
		a_vec[6].v = _mm256_loadu_pd( av[6] + 0*n_elem_per_reg );
		a_vec[7].v = _mm256_loadu_pd( av[7] + 0*n_elem_per_reg );
		
		a_vec[8].v = _mm256_loadu_pd( av[0] + 1*n_elem_per_reg );
		a_vec[9].v = _mm256_loadu_pd( av[1] + 1*n_elem_per_reg );
		a_vec[10].v = _mm256_loadu_pd( av[2] + 1*n_elem_per_reg );
		a_vec[11].v = _mm256_loadu_pd( av[3] + 1*n_elem_per_reg );
		a_vec[12].v = _mm256_loadu_pd( av[4] + 1*n_elem_per_reg );
		a_vec[13].v = _mm256_loadu_pd( av[5] + 1*n_elem_per_reg );
		a_vec[14].v = _mm256_loadu_pd( av[6] + 1*n_elem_per_reg );
		a_vec[15].v = _mm256_loadu_pd( av[7] + 1*n_elem_per_reg );

		// perform : y += alpha * x;
		yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[5].v, chiv[5].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[6].v, chiv[6].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[7].v, chiv[7].v, yv[0].v );

		yv[1].v = _mm256_fmadd_pd( a_vec[8].v, chiv[0].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[9].v, chiv[1].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[10].v, chiv[2].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[11].v, chiv[3].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[12].v, chiv[4].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[13].v, chiv[5].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[14].v, chiv[6].v, yv[1].v );
		yv[1].v = _mm256_fmadd_pd( a_vec[15].v, chiv[7].v, yv[1].v );

		// Store the output.
		_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );
		_mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1].v );

		y0 += n_elem_per_reg * n_iter_unroll[2];
		av[0] += n_elem_per_reg * n_iter_unroll[2];
		av[1] += n_elem_per_reg * n_iter_unroll[2];
		av[2] += n_elem_per_reg * n_iter_unroll[2];
		av[3] += n_elem_per_reg * n_iter_unroll[2];
		av[4] += n_elem_per_reg * n_iter_unroll[2];
		av[5] += n_elem_per_reg * n_iter_unroll[2];
		av[6] += n_elem_per_reg * n_iter_unroll[2];
		av[7] += n_elem_per_reg * n_iter_unroll[2];
	}

	// 4 elements of the result are computed per iteration
	for ( i = 0; i < m_viter[3]; ++i )
	{
		// Load the input values.
		yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

		a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
		a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
		a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
		a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
		a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );
		a_vec[5].v = _mm256_loadu_pd( av[5] + 0*n_elem_per_reg );
		a_vec[6].v = _mm256_loadu_pd( av[6] + 0*n_elem_per_reg );
		a_vec[7].v = _mm256_loadu_pd( av[7] + 0*n_elem_per_reg );

		// perform : y += alpha * x;
		yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[5].v, chiv[5].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[6].v, chiv[6].v, yv[0].v );
		yv[0].v = _mm256_fmadd_pd( a_vec[7].v, chiv[7].v, yv[0].v );

		// Store the output.
		_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );

		y0 += n_elem_per_reg;
		av[0] += n_elem_per_reg;
		av[1] += n_elem_per_reg;
		av[2] += n_elem_per_reg;
		av[3] += n_elem_per_reg;
		av[4] += n_elem_per_reg;
		av[5] += n_elem_per_reg;
		av[6] += n_elem_per_reg;
		av[7] += n_elem_per_reg;
	}

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < m_left ; ++i )
	{
		double       y0c = *y0;

		const double a0c = *av[0];
		const double a1c = *av[1];
		const double a2c = *av[2];
		const double a3c = *av[3];
		const double a4c = *av[4];
		const double a5c = *av[5];
		const double a6c = *av[6];
		const double a7c = *av[7];

		y0c += chi[0] * a0c;
		y0c += chi[1] * a1c;
		y0c += chi[2] * a2c;
		y0c += chi[3] * a3c;
		y0c += chi[4] * a4c;
		y0c += chi[5] * a5c;
		y0c += chi[6] * a6c;
		y0c += chi[7] * a7c;

		*y0 = y0c;

		av[0] += inca;
		av[1] += inca;
		av[2] += inca;
		av[3] += inca;
		av[4] += inca;
		av[5] += inca;
		av[6] += inca;
		av[7] += inca;
		y0 += incy;
	}
}
