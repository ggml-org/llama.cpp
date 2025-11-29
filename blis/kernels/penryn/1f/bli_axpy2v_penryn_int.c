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


void bli_daxpy2v_penryn_int
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   alphax,
       const void*   alphay,
       const void*   x, inc_t incx,
       const void*   y, inc_t incy,
             void*   z, inc_t incz,
       const cntx_t* cntx
     )
{
	const double*  restrict alphax_cast = alphax;
	const double*  restrict alphay_cast = alphay;
	const double*  restrict x_cast      = x;
	const double*  restrict y_cast      = y;
	      double*  restrict z_cast      = z;

	const dim_t             n_elem_per_reg = 2;
	const dim_t             n_iter_unroll  = 4;

	      dim_t             n_pre;
	      dim_t             n_run;
	      dim_t             n_left;

	      double            alphaxc, alphayc, x1c, y1c;

	      v2df_t            alphaxv, alphayv;
	      v2df_t            x1v, y1v, z1v;
	      v2df_t            x2v, y2v, z2v;

	      bool              use_ref = FALSE;


	if ( bli_zero_dim1( n ) ) return;

	n_pre = 0;

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( incx != 1 || incy != 1 || incz != 1 )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( ( siz_t )x, 16 ) ||
	          bli_is_unaligned_to( ( siz_t )y, 16 ) ||
	          bli_is_unaligned_to( ( siz_t )z, 16 ) )
	{
		use_ref = TRUE;

		if ( bli_is_unaligned_to( ( siz_t )x, 16 ) &&
		     bli_is_unaligned_to( ( siz_t )y, 16 ) &&
		     bli_is_unaligned_to( ( siz_t )z, 16 ) )
		{
			use_ref = FALSE;
			n_pre   = 1;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		#if 0
		axpy2v_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPY2V_KER, cntx );

		f
		(
		  conjx,
		  conjy,
		  n,
		  alphax,
		  alphay,
		  x, incx,
		  y, incy,
		  z, incz,
		  cntx
		);
		#endif
		bli_abort();
		return;
	}


	n_run       = ( n - n_pre ) / ( n_elem_per_reg * n_iter_unroll );
	n_left      = ( n - n_pre ) % ( n_elem_per_reg * n_iter_unroll );

	alphaxc = *alphax_cast;
	alphayc = *alphay_cast;

	const double* restrict x1 = x_cast;
	const double* restrict y1 = y_cast;
	      double* restrict z1 = z_cast;

	if ( n_pre == 1 )
	{
		x1c = *x1;
		y1c = *y1;

		*z1 += alphaxc * x1c +
		       alphayc * y1c;

		x1 += incx;
		y1 += incy;
		z1 += incz;
	}

	alphaxv.v = _mm_loaddup_pd( ( double* )alphax_cast );
	alphayv.v = _mm_loaddup_pd( ( double* )alphay_cast );

	for ( dim_t i = 0; i < n_run; ++i )
	{
		z1v.v = _mm_load_pd( ( double* )z1 + 0*n_elem_per_reg );
		x1v.v = _mm_load_pd( ( double* )x1 + 0*n_elem_per_reg );
		y1v.v = _mm_load_pd( ( double* )y1 + 0*n_elem_per_reg );

		z2v.v = _mm_load_pd( ( double* )z1 + 1*n_elem_per_reg );
		x2v.v = _mm_load_pd( ( double* )x1 + 1*n_elem_per_reg );
		y2v.v = _mm_load_pd( ( double* )y1 + 1*n_elem_per_reg );

		z1v.v += alphaxv.v * x1v.v;
		z1v.v += alphayv.v  * y1v.v;

		_mm_store_pd( ( double* )(z1 + 0*n_elem_per_reg ), z1v.v );

		z1v.v = _mm_load_pd( ( double* )z1 + 2*n_elem_per_reg );
		x1v.v = _mm_load_pd( ( double* )x1 + 2*n_elem_per_reg );
		y1v.v = _mm_load_pd( ( double* )y1 + 2*n_elem_per_reg );

		z2v.v += alphaxv.v * x2v.v;
		z2v.v += alphayv.v  * y2v.v;

		_mm_store_pd( ( double* )(z1 + 1*n_elem_per_reg ), z2v.v );

		z2v.v = _mm_load_pd( ( double* )z1 + 3*n_elem_per_reg );
		x2v.v = _mm_load_pd( ( double* )x1 + 3*n_elem_per_reg );
		y2v.v = _mm_load_pd( ( double* )y1 + 3*n_elem_per_reg );

		z1v.v += alphaxv.v * x1v.v;
		z1v.v += alphayv.v  * y1v.v;

		_mm_store_pd( ( double* )(z1 + 2*n_elem_per_reg ), z1v.v );

		z2v.v += alphaxv.v * x2v.v;
		z2v.v += alphayv.v  * y2v.v;

		_mm_store_pd( ( double* )(z1 + 3*n_elem_per_reg ), z2v.v );



		x1 += n_elem_per_reg * n_iter_unroll;
		y1 += n_elem_per_reg * n_iter_unroll;
		z1 += n_elem_per_reg * n_iter_unroll;
	}

	if ( n_left > 0 )
	{
		for ( dim_t i = 0; i < n_left; ++i )
		{
			x1c = *x1;
			y1c = *y1;

			*z1 += alphaxc * x1c +
			       alphayc * y1c;

			x1 += incx;
			y1 += incy;
			z1 += incz;
		}
	}
}
