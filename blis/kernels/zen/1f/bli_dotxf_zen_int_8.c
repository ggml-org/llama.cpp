/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2016 - 2018, Advanced Micro Devices, Inc.

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

void bli_sdotxf_zen_int_8
     (
             conj_t  conjat,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
       const void*   beta0,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const float* restrict alpha = alpha0;
	const float* restrict a     = a0;
	const float* restrict x     = x0;
	const float* restrict beta  = beta0;
	      float* restrict y     = y0;

	const dim_t fuse_fac       = 8;
	const dim_t n_elem_per_reg = 8;

	// If the b_n dimension is zero, y is empty and there is no computation.
	if ( bli_zero_dim1( b_n ) ) return;

	// If the m dimension is zero, or if alpha is zero, the computation
	// simplifies to updating y.
	if ( bli_zero_dim1( m ) || PASTEMAC(s,eq0)( *alpha ) )
	{
		scalv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_FLOAT, BLIS_SCALV_KER, cntx );

		f
		(
		  BLIS_NO_CONJUGATE,
		  b_n,
		  beta,
		  y, incy,
		  cntx
		);
		return;
	}

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over dotxv.
	if ( b_n != fuse_fac )
	{
		dotxv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_FLOAT, BLIS_DOTXV_KER, cntx );

		for ( dim_t i = 0; i < b_n; ++i )
		{
			const float* restrict a1   = a + (0  )*inca + (i  )*lda;
			const float* restrict x1   = x + (0  )*incx;
			      float* restrict psi1 = y + (i  )*incy;

			f
			(
			  conjat,
			  conjx,
			  m,
			  alpha,
			  a1, inca,
			  x1, incx,
			  beta,
			  psi1,
			  cntx
			);
		}
		return;
	}

	// At this point, we know that b_n is exactly equal to the fusing factor.
	// However, m may not be a multiple of the number of elements per vector.

	// Going forward, we handle two possible storage formats of A explicitly:
	// (1) A is stored by columns, or (2) A is stored by rows. Either case is
	// further split into two subproblems along the m dimension:
	// (a) a vectorized part, starting at m = 0 and ending at any 0 <= m' <= m.
	// (b) a scalar part, starting at m' and ending at m. If no vectorization
	//     is possible then m' == 0 and thus the scalar part is the entire
	//     problem. If 0 < m', then the a and x pointers and m variable will
	//     be adjusted accordingly for the second subproblem.
	// Note: since parts (b) for both (1) and (2) are so similar, they are
	// factored out into one code block after the following conditional, which
	// distinguishes between (1) and (2).

	// Intermediate variables to hold the completed dot products
    float rho0 = 0, rho1 = 0, rho2 = 0, rho3 = 0,
	      rho4 = 0, rho5 = 0, rho6 = 0, rho7 = 0;

	if ( inca == 1 && incx == 1 )
	{
		const dim_t n_iter_unroll = 1;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter = ( m ) / ( n_elem_per_reg * n_iter_unroll );

		// Set up pointers for x and the b_n columns of A (rows of A^T).
		const float* restrict xp0 = x;
		const float* restrict ap0 = a + 0*lda;
		const float* restrict ap1 = a + 1*lda;
		const float* restrict ap2 = a + 2*lda;
		const float* restrict ap3 = a + 3*lda;
		const float* restrict ap4 = a + 4*lda;
		const float* restrict ap5 = a + 5*lda;
		const float* restrict ap6 = a + 6*lda;
		const float* restrict ap7 = a + 7*lda;

		// Initialize b_n rho vector accumulators to zero.
		v8sf_t rho0v; rho0v.v = _mm256_setzero_ps();
		v8sf_t rho1v; rho1v.v = _mm256_setzero_ps();
		v8sf_t rho2v; rho2v.v = _mm256_setzero_ps();
		v8sf_t rho3v; rho3v.v = _mm256_setzero_ps();
		v8sf_t rho4v; rho4v.v = _mm256_setzero_ps();
		v8sf_t rho5v; rho5v.v = _mm256_setzero_ps();
		v8sf_t rho6v; rho6v.v = _mm256_setzero_ps();
		v8sf_t rho7v; rho7v.v = _mm256_setzero_ps();

		v8sf_t x0v;
		v8sf_t a0v, a1v, a2v, a3v, a4v, a5v, a6v, a7v;

		// If there are vectorized iterations, perform them with vector
		// instructions.
		for ( dim_t i = 0; i < m_viter; ++i )
		{
			// Load the input values.
			x0v.v = _mm256_loadu_ps( xp0 + 0*n_elem_per_reg );

			a0v.v = _mm256_loadu_ps( ap0 + 0*n_elem_per_reg );
			a1v.v = _mm256_loadu_ps( ap1 + 0*n_elem_per_reg );
			a2v.v = _mm256_loadu_ps( ap2 + 0*n_elem_per_reg );
			a3v.v = _mm256_loadu_ps( ap3 + 0*n_elem_per_reg );
			a4v.v = _mm256_loadu_ps( ap4 + 0*n_elem_per_reg );
			a5v.v = _mm256_loadu_ps( ap5 + 0*n_elem_per_reg );
			a6v.v = _mm256_loadu_ps( ap6 + 0*n_elem_per_reg );
			a7v.v = _mm256_loadu_ps( ap7 + 0*n_elem_per_reg );

			// perform: rho?v += a?v * x0v;
			rho0v.v = _mm256_fmadd_ps( a0v.v, x0v.v, rho0v.v );
			rho1v.v = _mm256_fmadd_ps( a1v.v, x0v.v, rho1v.v );
			rho2v.v = _mm256_fmadd_ps( a2v.v, x0v.v, rho2v.v );
			rho3v.v = _mm256_fmadd_ps( a3v.v, x0v.v, rho3v.v );
			rho4v.v = _mm256_fmadd_ps( a4v.v, x0v.v, rho4v.v );
			rho5v.v = _mm256_fmadd_ps( a5v.v, x0v.v, rho5v.v );
			rho6v.v = _mm256_fmadd_ps( a6v.v, x0v.v, rho6v.v );
			rho7v.v = _mm256_fmadd_ps( a7v.v, x0v.v, rho7v.v );

			xp0 += n_elem_per_reg * n_iter_unroll;
			ap0 += n_elem_per_reg * n_iter_unroll;
			ap1 += n_elem_per_reg * n_iter_unroll;
			ap2 += n_elem_per_reg * n_iter_unroll;
			ap3 += n_elem_per_reg * n_iter_unroll;
			ap4 += n_elem_per_reg * n_iter_unroll;
			ap5 += n_elem_per_reg * n_iter_unroll;
			ap6 += n_elem_per_reg * n_iter_unroll;
			ap7 += n_elem_per_reg * n_iter_unroll;
		}

#if 0
		rho0 += rho0v.f[0] + rho0v.f[1] + rho0v.f[2] + rho0v.f[3] +
		        rho0v.f[4] + rho0v.f[5] + rho0v.f[6] + rho0v.f[7];
		rho1 += rho1v.f[0] + rho1v.f[1] + rho1v.f[2] + rho1v.f[3] +
		        rho1v.f[4] + rho1v.f[5] + rho1v.f[6] + rho1v.f[7];
		rho2 += rho2v.f[0] + rho2v.f[1] + rho2v.f[2] + rho2v.f[3] +
		        rho2v.f[4] + rho2v.f[5] + rho2v.f[6] + rho2v.f[7];
		rho3 += rho3v.f[0] + rho3v.f[1] + rho3v.f[2] + rho3v.f[3] +
		        rho3v.f[4] + rho3v.f[5] + rho3v.f[6] + rho3v.f[7];
		rho4 += rho4v.f[0] + rho4v.f[1] + rho4v.f[2] + rho4v.f[3] +
		        rho4v.f[4] + rho4v.f[5] + rho4v.f[6] + rho4v.f[7];
		rho5 += rho5v.f[0] + rho5v.f[1] + rho5v.f[2] + rho5v.f[3] +
		        rho5v.f[4] + rho5v.f[5] + rho5v.f[6] + rho5v.f[7];
		rho6 += rho6v.f[0] + rho6v.f[1] + rho6v.f[2] + rho6v.f[3] +
		        rho6v.f[4] + rho6v.f[5] + rho6v.f[6] + rho6v.f[7];
		rho7 += rho7v.f[0] + rho7v.f[1] + rho7v.f[2] + rho7v.f[3] +
		        rho7v.f[4] + rho7v.f[5] + rho7v.f[6] + rho7v.f[7];
#else
		// Now we need to sum the elements within each vector.

		v8sf_t onev; onev.v = _mm256_set1_ps( 1.0f );

		// Sum the elements of a given rho?v by dotting it with 1. The '1' in
		// '0xf1' stores the sum of the upper four and lower four values to
		// the low elements of each lane: elements 4 and 0, respectively. (The
		// 'f' in '0xf1' means include all four elements of each lane in the
		// summation.)
		rho0v.v = _mm256_dp_ps( rho0v.v, onev.v, 0xf1 );
		rho1v.v = _mm256_dp_ps( rho1v.v, onev.v, 0xf1 );
		rho2v.v = _mm256_dp_ps( rho2v.v, onev.v, 0xf1 );
		rho3v.v = _mm256_dp_ps( rho3v.v, onev.v, 0xf1 );
		rho4v.v = _mm256_dp_ps( rho4v.v, onev.v, 0xf1 );
		rho5v.v = _mm256_dp_ps( rho5v.v, onev.v, 0xf1 );
		rho6v.v = _mm256_dp_ps( rho6v.v, onev.v, 0xf1 );
		rho7v.v = _mm256_dp_ps( rho7v.v, onev.v, 0xf1 );

		// Manually add the results from above to finish the sum.
		rho0    = rho0v.f[0] + rho0v.f[4];
		rho1    = rho1v.f[0] + rho1v.f[4];
		rho2    = rho2v.f[0] + rho2v.f[4];
		rho3    = rho3v.f[0] + rho3v.f[4];
		rho4    = rho4v.f[0] + rho4v.f[4];
		rho5    = rho5v.f[0] + rho5v.f[4];
		rho6    = rho6v.f[0] + rho6v.f[4];
		rho7    = rho7v.f[0] + rho7v.f[4];
#endif

		// Adjust for scalar subproblem.
		m -= n_elem_per_reg * n_iter_unroll * m_viter;
		a += n_elem_per_reg * n_iter_unroll * m_viter /* * inca */;
		x += n_elem_per_reg * n_iter_unroll * m_viter /* * incx */;
	}
	else if ( lda == 1 )
	{
		const dim_t n_iter_unroll = 4;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter = ( m ) / ( n_iter_unroll );

		// Initialize pointers for x and A.
		const float* restrict xp0 = x;
		const float* restrict ap0 = a;

		// Initialize rho vector accumulators to zero.
		v8sf_t rho0v; rho0v.v = _mm256_setzero_ps();
		v8sf_t rho1v; rho1v.v = _mm256_setzero_ps();
		v8sf_t rho2v; rho2v.v = _mm256_setzero_ps();
		v8sf_t rho3v; rho3v.v = _mm256_setzero_ps();

		v8sf_t x0v, x1v, x2v, x3v;
		v8sf_t a0v, a1v, a2v, a3v;

		for ( dim_t i = 0; i < m_viter; ++i )
		{
			// Load the input values.
			a0v.v = _mm256_loadu_ps( ap0 + 0*inca );
			a1v.v = _mm256_loadu_ps( ap0 + 1*inca );
			a2v.v = _mm256_loadu_ps( ap0 + 2*inca );
			a3v.v = _mm256_loadu_ps( ap0 + 3*inca );

			x0v.v = _mm256_broadcast_ss( xp0 + 0*incx );
			x1v.v = _mm256_broadcast_ss( xp0 + 1*incx );
			x2v.v = _mm256_broadcast_ss( xp0 + 2*incx );
			x3v.v = _mm256_broadcast_ss( xp0 + 3*incx );

			// perform : rho?v += a?v * x?v;
			rho0v.v = _mm256_fmadd_ps( a0v.v, x0v.v, rho0v.v );
			rho1v.v = _mm256_fmadd_ps( a1v.v, x1v.v, rho1v.v );
			rho2v.v = _mm256_fmadd_ps( a2v.v, x2v.v, rho2v.v );
			rho3v.v = _mm256_fmadd_ps( a3v.v, x3v.v, rho3v.v );

			xp0 += incx * n_iter_unroll;
			ap0 += inca * n_iter_unroll;
		}

		// Combine the 8 accumulators into one vector register.
		rho0v.v = _mm256_add_ps( rho0v.v, rho1v.v );
		rho2v.v = _mm256_add_ps( rho2v.v, rho3v.v );
		rho0v.v = _mm256_add_ps( rho0v.v, rho2v.v );

		// Write vector components to scalar values.
		rho0 = rho0v.f[0];
		rho1 = rho0v.f[1];
		rho2 = rho0v.f[2];
		rho3 = rho0v.f[3];
		rho4 = rho0v.f[4];
		rho5 = rho0v.f[5];
		rho6 = rho0v.f[6];
		rho7 = rho0v.f[7];

		// Adjust for scalar subproblem.
		m -= n_iter_unroll * m_viter;
		a += n_iter_unroll * m_viter * inca;
		x += n_iter_unroll * m_viter * incx;
	}
	else
	{
		// No vectorization possible; use scalar iterations for the entire
		// problem.
	}

	// Scalar edge case.
	{
		// Initialize pointers for x and the b_n columns of A (rows of A^T).
		const float* restrict xp0 = x;
		const float* restrict ap0 = a + 0*lda;
		const float* restrict ap1 = a + 1*lda;
		const float* restrict ap2 = a + 2*lda;
		const float* restrict ap3 = a + 3*lda;
		const float* restrict ap4 = a + 4*lda;
		const float* restrict ap5 = a + 5*lda;
		const float* restrict ap6 = a + 6*lda;
		const float* restrict ap7 = a + 7*lda;

		// If there are leftover iterations, perform them with scalar code.
		for ( dim_t i = 0; i < m ; ++i )
		{
			const float x0c = *xp0;

			const float a0c = *ap0;
			const float a1c = *ap1;
			const float a2c = *ap2;
			const float a3c = *ap3;
			const float a4c = *ap4;
			const float a5c = *ap5;
			const float a6c = *ap6;
			const float a7c = *ap7;

			rho0 += a0c * x0c;
			rho1 += a1c * x0c;
			rho2 += a2c * x0c;
			rho3 += a3c * x0c;
			rho4 += a4c * x0c;
			rho5 += a5c * x0c;
			rho6 += a6c * x0c;
			rho7 += a7c * x0c;

			xp0 += incx;
			ap0 += inca;
			ap1 += inca;
			ap2 += inca;
			ap3 += inca;
			ap4 += inca;
			ap5 += inca;
			ap6 += inca;
			ap7 += inca;
		}
	}

	// Now prepare the final rho values to output/accumulate back into
	// the y vector.

	v8sf_t rho0v, y0v;

	// Insert the scalar rho values into a single vector.
	rho0v.f[0] = rho0;
	rho0v.f[1] = rho1;
	rho0v.f[2] = rho2;
	rho0v.f[3] = rho3;
	rho0v.f[4] = rho4;
	rho0v.f[5] = rho5;
	rho0v.f[6] = rho6;
	rho0v.f[7] = rho7;

	// Broadcast the alpha scalar.
	v8sf_t alphav; alphav.v = _mm256_broadcast_ss( alpha );

	// We know at this point that alpha is nonzero; however, beta may still
	// be zero. If beta is indeed zero, we must overwrite y rather than scale
	// by beta (in case y contains NaN or Inf).
	if ( PASTEMAC(s,eq0)( *beta ) )
	{
		// Apply alpha to the accumulated dot product in rho:
		//   y := alpha * rho
		y0v.v = _mm256_mul_ps( alphav.v, rho0v.v );
	}
	else
	{
		// Broadcast the beta scalar.
		v8sf_t betav; betav.v = _mm256_broadcast_ss( beta );

		// Load y.
		if ( incy == 1 )
		{
			y0v.v = _mm256_loadu_ps( y + 0*n_elem_per_reg );
		}
		else
		{
			y0v.f[0] = *(y + 0*incy); y0v.f[1] = *(y + 1*incy);
			y0v.f[2] = *(y + 2*incy); y0v.f[3] = *(y + 3*incy);
			y0v.f[4] = *(y + 4*incy); y0v.f[5] = *(y + 5*incy);
			y0v.f[6] = *(y + 6*incy); y0v.f[7] = *(y + 7*incy);
		}

		// Apply beta to y and alpha to the accumulated dot product in rho:
		//   y := beta * y + alpha * rho
		y0v.v = _mm256_mul_ps( betav.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( alphav.v, rho0v.v, y0v.v );
	}

	// Store the output.
	if ( incy == 1 )
	{
		_mm256_storeu_ps( (y + 0*n_elem_per_reg), y0v.v );
	}
	else
	{
		*(y + 0*incy) = y0v.f[0]; *(y + 1*incy) = y0v.f[1];
		*(y + 2*incy) = y0v.f[2]; *(y + 3*incy) = y0v.f[3];
		*(y + 4*incy) = y0v.f[4]; *(y + 5*incy) = y0v.f[5];
		*(y + 6*incy) = y0v.f[6]; *(y + 7*incy) = y0v.f[7];
	}
}

// -----------------------------------------------------------------------------

void bli_ddotxf_zen_int_8
     (
             conj_t  conjat,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
       const void*   beta0,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* restrict alpha = alpha0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	const double* restrict beta  = beta0;
	      double* restrict y     = y0;

	const dim_t fuse_fac       = 8;
	const dim_t n_elem_per_reg = 4;

	// If the b_n dimension is zero, y is empty and there is no computation.
	if ( bli_zero_dim1( b_n ) ) return;

	// If the m dimension is zero, or if alpha is zero, the computation
	// simplifies to updating y.
	if ( bli_zero_dim1( m ) || PASTEMAC(d,eq0)( *alpha ) )
	{
		scalv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_SCALV_KER, cntx );

		f
		(
		  BLIS_NO_CONJUGATE,
		  b_n,
		  beta,
		  y, incy,
		  cntx
		);
		return;
	}

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over dotxv.
	if ( b_n != fuse_fac )
	{
		dotxv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_DOTXV_KER, cntx );

		for ( dim_t i = 0; i < b_n; ++i )
		{
			const double* restrict a1   = a + (0  )*inca + (i  )*lda;
			const double* restrict x1   = x + (0  )*incx;
			      double* restrict psi1 = y + (i  )*incy;

			f
			(
			  conjat,
			  conjx,
			  m,
			  alpha,
			  a1, inca,
			  x1, incx,
			  beta,
			  psi1,
			  cntx
			);
		}
		return;
	}

	// At this point, we know that b_n is exactly equal to the fusing factor.
	// However, m may not be a multiple of the number of elements per vector.

	// Going forward, we handle two possible storage formats of A explicitly:
	// (1) A is stored by columns, or (2) A is stored by rows. Either case is
	// further split into two subproblems along the m dimension:
	// (a) a vectorized part, starting at m = 0 and ending at any 0 <= m' <= m.
	// (b) a scalar part, starting at m' and ending at m. If no vectorization
	//     is possible then m' == 0 and thus the scalar part is the entire
	//     problem. If 0 < m', then the a and x pointers and m variable will
	//     be adjusted accordingly for the second subproblem.
	// Note: since parts (b) for both (1) and (2) are so similar, they are
	// factored out into one code block after the following conditional, which
	// distinguishes between (1) and (2).

	// Intermediate variables to hold the completed dot products
	double rho0 = 0, rho1 = 0, rho2 = 0, rho3 = 0,
	       rho4 = 0, rho5 = 0, rho6 = 0, rho7 = 0;

	if ( inca == 1 && incx == 1 )
	{
		const dim_t n_iter_unroll = 1;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter = ( m ) / ( n_elem_per_reg * n_iter_unroll );

		// Set up pointers for x and the b_n columns of A (rows of A^T).
		const double* restrict xp0 = x;
		const double* restrict ap0 = a + 0*lda;
		const double* restrict ap1 = a + 1*lda;
		const double* restrict ap2 = a + 2*lda;
		const double* restrict ap3 = a + 3*lda;
		const double* restrict ap4 = a + 4*lda;
		const double* restrict ap5 = a + 5*lda;
		const double* restrict ap6 = a + 6*lda;
		const double* restrict ap7 = a + 7*lda;

		// Initialize b_n rho vector accumulators to zero.
		v4df_t rho0v; rho0v.v = _mm256_setzero_pd();
		v4df_t rho1v; rho1v.v = _mm256_setzero_pd();
		v4df_t rho2v; rho2v.v = _mm256_setzero_pd();
		v4df_t rho3v; rho3v.v = _mm256_setzero_pd();
		v4df_t rho4v; rho4v.v = _mm256_setzero_pd();
		v4df_t rho5v; rho5v.v = _mm256_setzero_pd();
		v4df_t rho6v; rho6v.v = _mm256_setzero_pd();
		v4df_t rho7v; rho7v.v = _mm256_setzero_pd();

		v4df_t x0v;
		v4df_t a0v, a1v, a2v, a3v, a4v, a5v, a6v, a7v;

		// If there are vectorized iterations, perform them with vector
		// instructions.
		for ( dim_t i = 0; i < m_viter; ++i )
		{
			// Load the input values.
			x0v.v = _mm256_loadu_pd( xp0 + 0*n_elem_per_reg );

			a0v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
			a1v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
			a2v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );
			a3v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );
			a4v.v = _mm256_loadu_pd( ap4 + 0*n_elem_per_reg );
			a5v.v = _mm256_loadu_pd( ap5 + 0*n_elem_per_reg );
			a6v.v = _mm256_loadu_pd( ap6 + 0*n_elem_per_reg );
			a7v.v = _mm256_loadu_pd( ap7 + 0*n_elem_per_reg );

			// perform: rho?v += a?v * x0v;
			rho0v.v = _mm256_fmadd_pd( a0v.v, x0v.v, rho0v.v );
			rho1v.v = _mm256_fmadd_pd( a1v.v, x0v.v, rho1v.v );
			rho2v.v = _mm256_fmadd_pd( a2v.v, x0v.v, rho2v.v );
			rho3v.v = _mm256_fmadd_pd( a3v.v, x0v.v, rho3v.v );
			rho4v.v = _mm256_fmadd_pd( a4v.v, x0v.v, rho4v.v );
			rho5v.v = _mm256_fmadd_pd( a5v.v, x0v.v, rho5v.v );
			rho6v.v = _mm256_fmadd_pd( a6v.v, x0v.v, rho6v.v );
			rho7v.v = _mm256_fmadd_pd( a7v.v, x0v.v, rho7v.v );

			xp0 += n_elem_per_reg * n_iter_unroll;
			ap0 += n_elem_per_reg * n_iter_unroll;
			ap1 += n_elem_per_reg * n_iter_unroll;
			ap2 += n_elem_per_reg * n_iter_unroll;
			ap3 += n_elem_per_reg * n_iter_unroll;
			ap4 += n_elem_per_reg * n_iter_unroll;
			ap5 += n_elem_per_reg * n_iter_unroll;
			ap6 += n_elem_per_reg * n_iter_unroll;
			ap7 += n_elem_per_reg * n_iter_unroll;
		}

#if 0
		rho0 += rho0v.d[0] + rho0v.d[1] + rho0v.d[2] + rho0v.d[3];
		rho1 += rho1v.d[0] + rho1v.d[1] + rho1v.d[2] + rho1v.d[3];
		rho2 += rho2v.d[0] + rho2v.d[1] + rho2v.d[2] + rho2v.d[3];
		rho3 += rho3v.d[0] + rho3v.d[1] + rho3v.d[2] + rho3v.d[3];
		rho4 += rho4v.d[0] + rho4v.d[1] + rho4v.d[2] + rho4v.d[3];
		rho5 += rho5v.d[0] + rho5v.d[1] + rho5v.d[2] + rho5v.d[3];
		rho6 += rho6v.d[0] + rho6v.d[1] + rho6v.d[2] + rho6v.d[3];
		rho7 += rho7v.d[0] + rho7v.d[1] + rho7v.d[2] + rho7v.d[3];
#else
		// Sum the elements of a given rho?v. This computes the sum of
		// elements within lanes and stores the sum to both elements.
		rho0v.v = _mm256_hadd_pd( rho0v.v, rho0v.v );
		rho1v.v = _mm256_hadd_pd( rho1v.v, rho1v.v );
		rho2v.v = _mm256_hadd_pd( rho2v.v, rho2v.v );
		rho3v.v = _mm256_hadd_pd( rho3v.v, rho3v.v );
		rho4v.v = _mm256_hadd_pd( rho4v.v, rho4v.v );
		rho5v.v = _mm256_hadd_pd( rho5v.v, rho5v.v );
		rho6v.v = _mm256_hadd_pd( rho6v.v, rho6v.v );
		rho7v.v = _mm256_hadd_pd( rho7v.v, rho7v.v );

		// Manually add the results from above to finish the sum.
		rho0 = rho0v.d[0] + rho0v.d[2];
		rho1 = rho1v.d[0] + rho1v.d[2];
		rho2 = rho2v.d[0] + rho2v.d[2];
		rho3 = rho3v.d[0] + rho3v.d[2];
		rho4 = rho4v.d[0] + rho4v.d[2];
		rho5 = rho5v.d[0] + rho5v.d[2];
		rho6 = rho6v.d[0] + rho6v.d[2];
		rho7 = rho7v.d[0] + rho7v.d[2];
#endif
		// Adjust for scalar subproblem.
		m -= n_elem_per_reg * n_iter_unroll * m_viter;
		a += n_elem_per_reg * n_iter_unroll * m_viter /* * inca */;
		x += n_elem_per_reg * n_iter_unroll * m_viter /* * incx */;
	}
	else if ( lda == 1 )
	{
		const dim_t n_iter_unroll = 3;
		const dim_t n_reg_per_row = 2; // fuse_fac / n_elem_per_reg;

		// Use the unrolling factor and the number of elements per register
		// to compute the number of vectorized and leftover iterations.
		dim_t m_viter = ( m ) / ( n_reg_per_row * n_iter_unroll );

		// Initialize pointers for x and A.
		const double* restrict xp0 = x;
		const double* restrict ap0 = a;

		// Initialize rho vector accumulators to zero.
		v4df_t rho0v; rho0v.v = _mm256_setzero_pd();
		v4df_t rho1v; rho1v.v = _mm256_setzero_pd();
		v4df_t rho2v; rho2v.v = _mm256_setzero_pd();
		v4df_t rho3v; rho3v.v = _mm256_setzero_pd();
		v4df_t rho4v; rho4v.v = _mm256_setzero_pd();
		v4df_t rho5v; rho5v.v = _mm256_setzero_pd();

		v4df_t x0v, x1v, x2v;
		v4df_t a0v, a1v, a2v, a3v, a4v, a5v;

		for ( dim_t i = 0; i < m_viter; ++i )
		{
			// Load the input values.
			a0v.v = _mm256_loadu_pd( ap0 + 0*inca + 0*n_elem_per_reg );
			a1v.v = _mm256_loadu_pd( ap0 + 0*inca + 1*n_elem_per_reg );
			a2v.v = _mm256_loadu_pd( ap0 + 1*inca + 0*n_elem_per_reg );
			a3v.v = _mm256_loadu_pd( ap0 + 1*inca + 1*n_elem_per_reg );
			a4v.v = _mm256_loadu_pd( ap0 + 2*inca + 0*n_elem_per_reg );
			a5v.v = _mm256_loadu_pd( ap0 + 2*inca + 1*n_elem_per_reg );

			x0v.v = _mm256_broadcast_sd( xp0 + 0*incx );
			x1v.v = _mm256_broadcast_sd( xp0 + 1*incx );
			x2v.v = _mm256_broadcast_sd( xp0 + 2*incx );

			// perform : rho?v += a?v * x?v;
			rho0v.v = _mm256_fmadd_pd( a0v.v, x0v.v, rho0v.v );
			rho1v.v = _mm256_fmadd_pd( a1v.v, x0v.v, rho1v.v );
			rho2v.v = _mm256_fmadd_pd( a2v.v, x1v.v, rho2v.v );
			rho3v.v = _mm256_fmadd_pd( a3v.v, x1v.v, rho3v.v );
			rho4v.v = _mm256_fmadd_pd( a4v.v, x2v.v, rho4v.v );
			rho5v.v = _mm256_fmadd_pd( a5v.v, x2v.v, rho5v.v );

			xp0 += incx * n_iter_unroll;
			ap0 += inca * n_iter_unroll;
		}

		// Combine the 8 accumulators into one vector register.
		rho0v.v = _mm256_add_pd( rho0v.v, rho2v.v );
		rho0v.v = _mm256_add_pd( rho0v.v, rho4v.v );
		rho1v.v = _mm256_add_pd( rho1v.v, rho3v.v );
		rho1v.v = _mm256_add_pd( rho1v.v, rho5v.v );

		// Write vector components to scalar values.
		rho0 = rho0v.d[0];
		rho1 = rho0v.d[1];
		rho2 = rho0v.d[2];
		rho3 = rho0v.d[3];
		rho4 = rho1v.d[0];
		rho5 = rho1v.d[1];
		rho6 = rho1v.d[2];
		rho7 = rho1v.d[3];

		// Adjust for scalar subproblem.
		m -= n_iter_unroll * m_viter;
		a += n_iter_unroll * m_viter * inca;
		x += n_iter_unroll * m_viter * incx;
	}
	else
	{
		// No vectorization possible; use scalar iterations for the entire
		// problem.
	}

	// Scalar edge case.
	{
		// Initialize pointers for x and the b_n columns of A (rows of A^T).
		const double* restrict xp0 = x;
		const double* restrict ap0 = a + 0*lda;
		const double* restrict ap1 = a + 1*lda;
		const double* restrict ap2 = a + 2*lda;
		const double* restrict ap3 = a + 3*lda;
		const double* restrict ap4 = a + 4*lda;
		const double* restrict ap5 = a + 5*lda;
		const double* restrict ap6 = a + 6*lda;
		const double* restrict ap7 = a + 7*lda;

		// If there are leftover iterations, perform them with scalar code.
		for ( dim_t i = 0; i < m ; ++i )
		{
			const double x0c = *xp0;

			const double a0c = *ap0;
			const double a1c = *ap1;
			const double a2c = *ap2;
			const double a3c = *ap3;
			const double a4c = *ap4;
			const double a5c = *ap5;
			const double a6c = *ap6;
			const double a7c = *ap7;

			rho0 += a0c * x0c;
			rho1 += a1c * x0c;
			rho2 += a2c * x0c;
			rho3 += a3c * x0c;
			rho4 += a4c * x0c;
			rho5 += a5c * x0c;
			rho6 += a6c * x0c;
			rho7 += a7c * x0c;

			xp0 += incx;
			ap0 += inca;
			ap1 += inca;
			ap2 += inca;
			ap3 += inca;
			ap4 += inca;
			ap5 += inca;
			ap6 += inca;
			ap7 += inca;
		}
	}

	// Now prepare the final rho values to output/accumulate back into
	// the y vector.

	v4df_t rho0v, rho1v, y0v, y1v;

	// Insert the scalar rho values into a single vector.
	rho0v.d[0] = rho0;
	rho0v.d[1] = rho1;
	rho0v.d[2] = rho2;
	rho0v.d[3] = rho3;
	rho1v.d[0] = rho4;
	rho1v.d[1] = rho5;
	rho1v.d[2] = rho6;
	rho1v.d[3] = rho7;

	// Broadcast the alpha scalar.
	v4df_t alphav; alphav.v = _mm256_broadcast_sd( alpha );

	// We know at this point that alpha is nonzero; however, beta may still
	// be zero. If beta is indeed zero, we must overwrite y rather than scale
	// by beta (in case y contains NaN or Inf).
	if ( PASTEMAC(d,eq0)( *beta ) )
	{
		// Apply alpha to the accumulated dot product in rho:
		//   y := alpha * rho
		y0v.v = _mm256_mul_pd( alphav.v, rho0v.v );
		y1v.v = _mm256_mul_pd( alphav.v, rho1v.v );
	}
	else
	{
		// Broadcast the beta scalar.
		v4df_t betav; betav.v = _mm256_broadcast_sd( beta );

		// Load y.
		if ( incy == 1 )
		{
			y0v.v = _mm256_loadu_pd( y + 0*n_elem_per_reg );
			y1v.v = _mm256_loadu_pd( y + 1*n_elem_per_reg );
		}
		else
		{
			y0v.d[0] = *(y + 0*incy); y0v.d[1] = *(y + 1*incy);
			y0v.d[2] = *(y + 2*incy); y0v.d[3] = *(y + 3*incy);
			y1v.d[0] = *(y + 4*incy); y1v.d[1] = *(y + 5*incy);
			y1v.d[2] = *(y + 6*incy); y1v.d[3] = *(y + 7*incy);
		}

		// Apply beta to y and alpha to the accumulated dot product in rho:
		//   y := beta * y + alpha * rho
		y0v.v = _mm256_mul_pd( betav.v, y0v.v );
		y1v.v = _mm256_mul_pd( betav.v, y1v.v );
		y0v.v = _mm256_fmadd_pd( alphav.v, rho0v.v, y0v.v );
		y1v.v = _mm256_fmadd_pd( alphav.v, rho1v.v, y1v.v );
	}

	if ( incy == 1 )
	{
		// Store the output.
		_mm256_storeu_pd( (y + 0*n_elem_per_reg), y0v.v );
		_mm256_storeu_pd( (y + 1*n_elem_per_reg), y1v.v );
	}
	else
	{
		*(y + 0*incy) = y0v.d[0]; *(y + 1*incy) = y0v.d[1];
		*(y + 2*incy) = y0v.d[2]; *(y + 3*incy) = y0v.d[3];
		*(y + 4*incy) = y1v.d[0]; *(y + 5*incy) = y1v.d[1];
		*(y + 6*incy) = y1v.d[2]; *(y + 7*incy) = y1v.d[3];
	}
}

