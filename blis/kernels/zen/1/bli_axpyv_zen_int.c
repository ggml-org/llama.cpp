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

void bli_saxpyv_zen_int
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const float*     alpha = alpha0;
	const float*     x     = x0;
	      float*     y     = y0;

	const dim_t      n_elem_per_reg = 8;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	v8sf_t           alphav;
	v8sf_t           x0v, x1v, x2v, x3v;
	v8sf_t           y0v, y1v, y2v, y3v;

	// If the vector dimension is zero, or if alpha is zero, return early.
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
	      float* restrict yp = y;

	// Broadcast the alpha scalar to all elements of a vector register.
	alphav.v = _mm256_broadcast_ss( alpha );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	for ( i = 0; i < n_viter; ++i )
	{
		// Load the input values.
		y0v.v = _mm256_loadu_ps( yp + 0*n_elem_per_reg );
		x0v.v = _mm256_loadu_ps( xp + 0*n_elem_per_reg );

		y1v.v = _mm256_loadu_ps( yp + 1*n_elem_per_reg );
		x1v.v = _mm256_loadu_ps( xp + 1*n_elem_per_reg );

		y2v.v = _mm256_loadu_ps( yp + 2*n_elem_per_reg );
		x2v.v = _mm256_loadu_ps( xp + 2*n_elem_per_reg );

		y3v.v = _mm256_loadu_ps( yp + 3*n_elem_per_reg );
		x3v.v = _mm256_loadu_ps( xp + 3*n_elem_per_reg );

		// perform : y += alpha * x;
		y0v.v = _mm256_fmadd_ps( alphav.v, x0v.v, y0v.v );
		y1v.v = _mm256_fmadd_ps( alphav.v, x1v.v, y1v.v );
		y2v.v = _mm256_fmadd_ps( alphav.v, x2v.v, y2v.v );
		y3v.v = _mm256_fmadd_ps( alphav.v, x3v.v, y3v.v );

		// Store the output.
		_mm256_storeu_ps( (yp + 0*n_elem_per_reg), y0v.v );
		_mm256_storeu_ps( (yp + 1*n_elem_per_reg), y1v.v );
		_mm256_storeu_ps( (yp + 2*n_elem_per_reg), y2v.v );
		_mm256_storeu_ps( (yp + 3*n_elem_per_reg), y3v.v );

		xp += n_elem_per_reg * n_iter_unroll;
		yp += n_elem_per_reg * n_iter_unroll;
	}

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// as soon as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	const float alphac = *alpha;

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const float x0c = *xp;

		*yp += alphac * x0c;

		xp += incx;
		yp += incy;
	}
}

// -----------------------------------------------------------------------------

void bli_daxpyv_zen_int
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double*     alpha = alpha0;
	const double*     x     = x0;
	      double*     y     = y0;

	const dim_t       n_elem_per_reg = 4;
	const dim_t       n_iter_unroll  = 4;

	dim_t             i;
	dim_t             n_viter;
	dim_t             n_left;

	v4df_t            alphav;
	v4df_t            x0v, x1v, x2v, x3v;
	v4df_t            y0v, y1v, y2v, y3v;

	// If the vector dimension is zero, or if alpha is zero, return early.
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
	      double* restrict yp = y;

	// Broadcast the alpha scalar to all elements of a vector register.
	alphav.v = _mm256_broadcast_sd( alpha );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	for ( i = 0; i < n_viter; ++i )
	{
		// Load the input values.
		y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
		x0v.v = _mm256_loadu_pd( xp + 0*n_elem_per_reg );

		y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
		x1v.v = _mm256_loadu_pd( xp + 1*n_elem_per_reg );

		y2v.v = _mm256_loadu_pd( yp + 2*n_elem_per_reg );
		x2v.v = _mm256_loadu_pd( xp + 2*n_elem_per_reg );

		y3v.v = _mm256_loadu_pd( yp + 3*n_elem_per_reg );
		x3v.v = _mm256_loadu_pd( xp + 3*n_elem_per_reg );

		// perform : y += alpha * x;
		y0v.v = _mm256_fmadd_pd( alphav.v, x0v.v, y0v.v );
		y1v.v = _mm256_fmadd_pd( alphav.v, x1v.v, y1v.v );
		y2v.v = _mm256_fmadd_pd( alphav.v, x2v.v, y2v.v );
		y3v.v = _mm256_fmadd_pd( alphav.v, x3v.v, y3v.v );

		// Store the output.
		_mm256_storeu_pd( (yp + 0*n_elem_per_reg), y0v.v );
		_mm256_storeu_pd( (yp + 1*n_elem_per_reg), y1v.v );
		_mm256_storeu_pd( (yp + 2*n_elem_per_reg), y2v.v );
		_mm256_storeu_pd( (yp + 3*n_elem_per_reg), y3v.v );

		xp += n_elem_per_reg * n_iter_unroll;
		yp += n_elem_per_reg * n_iter_unroll;
	}

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// as soon as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	const double alphac = *alpha;

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const double x0c = *xp;

		*yp += alphac * x0c;

		xp += incx;
		yp += incy;
	}
}

