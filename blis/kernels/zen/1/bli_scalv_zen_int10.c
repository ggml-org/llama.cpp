/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2017 - 2022, Advanced Micro Devices, Inc.
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

void bli_sscalv_zen_int10
     (
             conj_t  conjalpha,
             dim_t   n,
       const void*   alpha0,
             void*   x0, inc_t incx,
       const cntx_t* cntx
     )
{
	const float*     alpha = alpha0;
	      float*     x     = x0;

	const dim_t      n_elem_per_reg = 8;

	dim_t            i;

	float*  restrict xp;

	__m256           alphav;
	__m256           xv[10];
	__m256           zv[10];

	// If the vector dimension is zero, or if alpha is unit, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(s,eq1)( *alpha ) ) return;

	// If alpha is zero, use setv.
	if ( PASTEMAC(s,eq0)( *alpha ) )
	{
		if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

		void*       zero = bli_s0;
		setv_ker_ft f    = bli_cntx_get_ukr_dt( BLIS_FLOAT, BLIS_SETV_KER, cntx );

		f
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  x, incx,
		  cntx
		);

		return;
	}

	// Initialize local pointers.
	xp = x;

	if ( incx == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_ss( alpha );

		for ( i = 0; (i + 79) < n; i += 80 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( xp + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_ps( xp + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_ps( xp + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_ps( xp + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_ps( xp + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_ps( xp + 9*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );
			zv[2] = _mm256_mul_ps( alphav, xv[2] );
			zv[3] = _mm256_mul_ps( alphav, xv[3] );
			zv[4] = _mm256_mul_ps( alphav, xv[4] );
			zv[5] = _mm256_mul_ps( alphav, xv[5] );
			zv[6] = _mm256_mul_ps( alphav, xv[6] );
			zv[7] = _mm256_mul_ps( alphav, xv[7] );
			zv[8] = _mm256_mul_ps( alphav, xv[8] );
			zv[9] = _mm256_mul_ps( alphav, xv[9] );

			// Store the output.
			_mm256_storeu_ps( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (xp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (xp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (xp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_ps( (xp + 4*n_elem_per_reg), zv[4] );
			_mm256_storeu_ps( (xp + 5*n_elem_per_reg), zv[5] );
			_mm256_storeu_ps( (xp + 6*n_elem_per_reg), zv[6] );
			_mm256_storeu_ps( (xp + 7*n_elem_per_reg), zv[7] );
			_mm256_storeu_ps( (xp + 8*n_elem_per_reg), zv[8] );
			_mm256_storeu_ps( (xp + 9*n_elem_per_reg), zv[9] );

			xp += 10*n_elem_per_reg;
		}

		for ( ; (i + 39) < n; i += 40 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( xp + 4*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );
			zv[2] = _mm256_mul_ps( alphav, xv[2] );
			zv[3] = _mm256_mul_ps( alphav, xv[3] );
			zv[4] = _mm256_mul_ps( alphav, xv[4] );

			// Store the output.
			_mm256_storeu_ps( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (xp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (xp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (xp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_ps( (xp + 4*n_elem_per_reg), zv[4] );

			xp += 5*n_elem_per_reg;
		}

		for ( ; (i + 31) < n; i += 32 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( xp + 3*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );
			zv[2] = _mm256_mul_ps( alphav, xv[2] );
			zv[3] = _mm256_mul_ps( alphav, xv[3] );

			// Store the output.
			_mm256_storeu_ps( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (xp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (xp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (xp + 3*n_elem_per_reg), zv[3] );

			xp += 4*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );

			// Store the output.
			_mm256_storeu_ps( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (xp + 1*n_elem_per_reg), zv[1] );

			xp += 2*n_elem_per_reg;
		}

		for ( ; (i + 7) < n; i += 8 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );

			// Store the output.
			_mm256_storeu_ps( (xp + 0*n_elem_per_reg), zv[0] );

			xp += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			*xp *= *alpha;

			xp += 1;
		}
	}
	else
	{
		const float alphac = *alpha;

		for ( i = 0; i < n; ++i )
		{
			*xp *= alphac;

			xp += incx;
		}
	}
}

// -----------------------------------------------------------------------------

void bli_dscalv_zen_int10
     (
             conj_t  conjalpha,
             dim_t   n,
       const void*   alpha0,
             void*   x0, inc_t incx,
       const cntx_t* cntx
     )
{
	const double*    alpha = alpha0;
	      double*    x     = x0;

	const dim_t      n_elem_per_reg = 4;

	dim_t            i;

	double* restrict xp;

	__m256d          alphav;
	__m256d          xv[10];
	__m256d          zv[10];

	// If the vector dimension is zero, or if alpha is unit, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(d,eq1)( *alpha ) ) return;

	// If alpha is zero, use setv.
	if ( PASTEMAC(d,eq0)( *alpha ) )
	{
		if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

		void*       zero = bli_d0;
		setv_ker_ft f    = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_SETV_KER, cntx );

		f
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  x, incx,
		  cntx
		);

		return;
	}

	// Initialize local pointers.
	xp = x;

	if ( incx == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_sd( alpha );

		for ( i = 0; (i + 39) < n; i += 40 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( xp + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( xp + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( xp + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( xp + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_pd( xp + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_pd( xp + 9*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );
			zv[2] = _mm256_mul_pd( alphav, xv[2] );
			zv[3] = _mm256_mul_pd( alphav, xv[3] );
			zv[4] = _mm256_mul_pd( alphav, xv[4] );
			zv[5] = _mm256_mul_pd( alphav, xv[5] );
			zv[6] = _mm256_mul_pd( alphav, xv[6] );
			zv[7] = _mm256_mul_pd( alphav, xv[7] );
			zv[8] = _mm256_mul_pd( alphav, xv[8] );
			zv[9] = _mm256_mul_pd( alphav, xv[9] );

			// Store the output.
			_mm256_storeu_pd( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (xp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (xp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (xp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_pd( (xp + 4*n_elem_per_reg), zv[4] );
			_mm256_storeu_pd( (xp + 5*n_elem_per_reg), zv[5] );
			_mm256_storeu_pd( (xp + 6*n_elem_per_reg), zv[6] );
			_mm256_storeu_pd( (xp + 7*n_elem_per_reg), zv[7] );
			_mm256_storeu_pd( (xp + 8*n_elem_per_reg), zv[8] );
			_mm256_storeu_pd( (xp + 9*n_elem_per_reg), zv[9] );

			xp += 10*n_elem_per_reg;
		}

		for ( ; (i + 19) < n; i += 20 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( xp + 4*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );
			zv[2] = _mm256_mul_pd( alphav, xv[2] );
			zv[3] = _mm256_mul_pd( alphav, xv[3] );
			zv[4] = _mm256_mul_pd( alphav, xv[4] );

			// Store the output.
			_mm256_storeu_pd( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (xp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (xp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (xp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_pd( (xp + 4*n_elem_per_reg), zv[4] );

			xp += 5*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( xp + 3*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );
			zv[2] = _mm256_mul_pd( alphav, xv[2] );
			zv[3] = _mm256_mul_pd( alphav, xv[3] );

			// Store the output.
			_mm256_storeu_pd( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (xp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (xp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (xp + 3*n_elem_per_reg), zv[3] );

			xp += 4*n_elem_per_reg;
		}

		for ( ; (i + 7) < n; i += 8 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );

			// Store the output.
			_mm256_storeu_pd( (xp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (xp + 1*n_elem_per_reg), zv[1] );

			xp += 2*n_elem_per_reg;
		}

		for ( ; (i + 3) < n; i += 4 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );

			// Store the output.
			_mm256_storeu_pd( (xp + 0*n_elem_per_reg), zv[0] );

			xp += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			*xp *= *alpha;

			xp += 1;
		}
	}
	else
	{
		const double alphac = *alpha;

		for ( i = 0; i < n; ++i )
		{
			*xp *= alphac;

			xp += incx;
		}
	}
}

// -----------------------------------------------------------------------------

//
// NOTE: This function definition is provided as a placeholder in order to allow
// function names of scalv kernels to be hard-coded in bli_gemv_unf_var2_amd.c.
//

void bli_cscalv_zen_int10
     (
             conj_t  conjalpha,
             dim_t   n,
       const void*   alpha,
             void*   x, inc_t incx,
       const cntx_t* cntx
     )
{
	scalv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_SCOMPLEX, BLIS_SCALV_KER, cntx );

	f
	(
	  conjalpha,
	  n,
	  alpha,
	  x, incx,
	  cntx
	);
}
