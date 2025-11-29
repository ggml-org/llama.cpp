/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#include "pmmintrin.h"
#include "blis.h"


typedef union
{
	__m128d v;
	double  d[2];
} v2df_t;


void bli_daxpyv_penryn_int
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double*  restrict alpha_cast = alpha0;
	const double*  restrict x_cast     = x0;
	      double*  restrict y_cast     = y0;

	const dim_t             n_elem_per_reg = 2;
	const dim_t             n_iter_unroll  = 4;

	      dim_t             n_pre;
	      dim_t             n_run;
	      dim_t             n_left;

	      double            alpha1c, x1c;

	      v2df_t            alpha1v;
	      v2df_t            x1v, x2v, x3v, x4v;
	      v2df_t            y1v, y2v, y3v, y4v;

	      bool              use_ref = FALSE;


	if ( bli_zero_dim1( n ) ) return;

	n_pre = 0;

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( incx != 1 || incy != 1 )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( ( siz_t )x_cast, 16 ) ||
	          bli_is_unaligned_to( ( siz_t )y_cast, 16 ) )
	{
		use_ref = TRUE;

		if ( bli_is_unaligned_to( ( siz_t )x_cast, 16 ) &&
		     bli_is_unaligned_to( ( siz_t )y_cast, 16 ) )
		{
			use_ref = FALSE;
			n_pre   = 1;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

		f
		(
		  conjx,
		  n,
		  alpha0,
		  x0, incx,
		  y0, incy,
		  cntx
		);
		return;
	}


	n_run       = ( n - n_pre ) / ( n_elem_per_reg * n_iter_unroll );
	n_left      = ( n - n_pre ) % ( n_elem_per_reg * n_iter_unroll );

	alpha1c = *alpha_cast;

	const double* restrict x1 = x_cast;
	      double* restrict y1 = y_cast;

	if ( n_pre == 1 )
	{
		x1c = *x1;

		*y1 += alpha1c * x1c;

		x1 += incx;
		y1 += incy;
	}

	alpha1v.v = _mm_loaddup_pd( ( double* )&alpha1c );

	for ( dim_t i = 0; i < n_run; ++i )
	{
		y1v.v = _mm_load_pd( ( double* )y1 );
		x1v.v = _mm_load_pd( ( double* )x1 );

		y1v.v += alpha1v.v * x1v.v;

		_mm_store_pd( ( double* )(y1    ), y1v.v );

		y2v.v = _mm_load_pd( ( double* )(y1 + 2) );
		x2v.v = _mm_load_pd( ( double* )(x1 + 2) );

		y2v.v += alpha1v.v * x2v.v;

		_mm_store_pd( ( double* )(y1 + 2), y2v.v );

		y3v.v = _mm_load_pd( ( double* )(y1 + 4) );
		x3v.v = _mm_load_pd( ( double* )(x1 + 4) );

		y3v.v += alpha1v.v * x3v.v;

		_mm_store_pd( ( double* )(y1 + 4), y3v.v );

		y4v.v = _mm_load_pd( ( double* )(y1 + 6) );
		x4v.v = _mm_load_pd( ( double* )(x1 + 6) );

		y4v.v += alpha1v.v * x4v.v;

		_mm_store_pd( ( double* )(y1 + 6), y4v.v );


		x1 += n_elem_per_reg * n_iter_unroll;
		y1 += n_elem_per_reg * n_iter_unroll;
	}

	if ( n_left > 0 )
	{
		for ( dim_t i = 0; i < n_left; ++i )
		{
			x1c = *x1;

			*y1 += alpha1c * x1c;

			x1 += incx;
			y1 += incy;
		}
	}
}
