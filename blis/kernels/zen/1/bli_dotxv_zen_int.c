/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2018, The University of Texas at Austin

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

void bli_sdotxv_zen_int
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
       const void*   y0, inc_t incy,
       const void*   beta0,
             void*   rho0,
       const cntx_t* cntx
     )
{
	const float*  alpha = alpha0;
	const float*  x     = x0;
	const float*  y     = y0;
	const float*  beta  = beta0;
	      float*  rho   = rho0;

	const dim_t      n_elem_per_reg = 8;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	float            rho_l;

	v8sf_t           rho0v, rho1v, rho2v, rho3v;
	v8sf_t           x0v, y0v;
	v8sf_t           x1v, y1v;
	v8sf_t           x2v, y2v;
	v8sf_t           x3v, y3v;

	// If beta is zero, initialize rho1 to zero instead of scaling
	// rho by beta (in case rho contains NaN or Inf).
	if ( PASTEMAC(s,eq0)( *beta ) )
	{
		PASTEMAC(s,set0s)( *rho );
	}
	else
	{
		PASTEMAC(s,scals)( *beta, *rho );
	}

	// If the vector dimension is zero, output rho and return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(s,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	const float* restrict xp = x;
	const float* restrict yp = y;

	// Initialize the unrolled iterations' rho vectors to zero.
	rho0v.v = _mm256_setzero_ps();
	rho1v.v = _mm256_setzero_ps();
	rho2v.v = _mm256_setzero_ps();
	rho3v.v = _mm256_setzero_ps();

	for ( i = 0; i < n_viter; ++i )
	{
		// Load the x and y input vector elements.
		x0v.v = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
		y0v.v = _mm256_loadu_ps( yp + 0*n_elem_per_reg );

		x1v.v = _mm256_loadu_ps( xp + 1*n_elem_per_reg );
		y1v.v = _mm256_loadu_ps( yp + 1*n_elem_per_reg );

		x2v.v = _mm256_loadu_ps( xp + 2*n_elem_per_reg );
		y2v.v = _mm256_loadu_ps( yp + 2*n_elem_per_reg );

		x3v.v = _mm256_loadu_ps( xp + 3*n_elem_per_reg );
		y3v.v = _mm256_loadu_ps( yp + 3*n_elem_per_reg );

		// Compute the element-wise product of the x and y vectors,
		// storing in the corresponding rho vectors.
		rho0v.v = _mm256_fmadd_ps( x0v.v, y0v.v, rho0v.v );
		rho1v.v = _mm256_fmadd_ps( x1v.v, y1v.v, rho1v.v );
		rho2v.v = _mm256_fmadd_ps( x2v.v, y2v.v, rho2v.v );
		rho3v.v = _mm256_fmadd_ps( x3v.v, y3v.v, rho3v.v );

		xp += ( n_elem_per_reg * n_iter_unroll );
		yp += ( n_elem_per_reg * n_iter_unroll );
	}

	// Accumulate the unrolled rho vectors into a single vector.
	rho0v.v += rho1v.v;
	rho0v.v += rho2v.v;
	rho0v.v += rho3v.v;

	// Accumulate the final rho vector into a single scalar result.
	rho_l = rho0v.f[0] + rho0v.f[1] + rho0v.f[2] + rho0v.f[3] +
	        rho0v.f[4] + rho0v.f[5] + rho0v.f[6] + rho0v.f[7];

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// as soon as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const float x0c = *xp;
		const float y0c = *yp;

		rho_l += x0c * y0c;

		xp += incx;
		yp += incy;
	}

	// Accumulate the final result into the output variable.
	PASTEMAC(s,axpys)( *alpha, rho_l, *rho );
}

// -----------------------------------------------------------------------------

void bli_ddotxv_zen_int
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
       const void*   y0, inc_t incy,
       const void*   beta0,
             void*   rho0,
       const cntx_t* cntx
     )
{
	const double*  alpha = alpha0;
	const double*  x     = x0;
	const double*  y     = y0;
	const double*  beta  = beta0;
	      double*  rho   = rho0;

	const dim_t      n_elem_per_reg = 4;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	double           rho_l;

	v4df_t           rho0v, rho1v, rho2v, rho3v;
	v4df_t           x0v, y0v;
	v4df_t           x1v, y1v;
	v4df_t           x2v, y2v;
	v4df_t           x3v, y3v;

	// If beta is zero, initialize rho1 to zero instead of scaling
	// rho by beta (in case rho contains NaN or Inf).
	if ( PASTEMAC(d,eq0)( *beta ) )
	{
		PASTEMAC(d,set0s)( *rho );
	}
	else
	{
		PASTEMAC(d,scals)( *beta, *rho );
	}

	// If the vector dimension is zero, output rho and return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(d,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	const double* restrict xp = x;
	const double* restrict yp = y;

	// Initialize the unrolled iterations' rho vectors to zero.
	rho0v.v = _mm256_setzero_pd();
	rho1v.v = _mm256_setzero_pd();
	rho2v.v = _mm256_setzero_pd();
	rho3v.v = _mm256_setzero_pd();

	for ( i = 0; i < n_viter; ++i )
	{
		// Load the x and y input vector elements.
		x0v.v = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
		y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );

		x1v.v = _mm256_loadu_pd( xp + 1*n_elem_per_reg );
		y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );

		x2v.v = _mm256_loadu_pd( xp + 2*n_elem_per_reg );
		y2v.v = _mm256_loadu_pd( yp + 2*n_elem_per_reg );

		x3v.v = _mm256_loadu_pd( xp + 3*n_elem_per_reg );
		y3v.v = _mm256_loadu_pd( yp + 3*n_elem_per_reg );

		// Compute the element-wise product of the x and y vectors,
		// storing in the corresponding rho vectors.
		rho0v.v = _mm256_fmadd_pd( x0v.v, y0v.v, rho0v.v );
		rho1v.v = _mm256_fmadd_pd( x1v.v, y1v.v, rho1v.v );
		rho2v.v = _mm256_fmadd_pd( x2v.v, y2v.v, rho2v.v );
		rho3v.v = _mm256_fmadd_pd( x3v.v, y3v.v, rho3v.v );

		xp += ( n_elem_per_reg * n_iter_unroll );
		yp += ( n_elem_per_reg * n_iter_unroll );
	}

	// Accumulate the unrolled rho vectors into a single vector.
	rho0v.v += rho1v.v;
	rho0v.v += rho2v.v;
	rho0v.v += rho3v.v;

	// Accumulate the final rho vector into a single scalar result.
	rho_l = rho0v.d[0] + rho0v.d[1] + rho0v.d[2] + rho0v.d[3];

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// as soon as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const double x0c = *xp;
		const double y0c = *yp;

		rho_l += x0c * y0c;

		xp += incx;
		yp += incy;
	}

	// Accumulate the final result into the output variable.
	PASTEMAC(d,axpys)( *alpha, rho_l, *rho );
}

