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


void bli_ddotv_penryn_int
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   x0, inc_t incx,
       const void*   y0, inc_t incy,
             void*   rho0,
       const cntx_t* cntx
     )
{
	const double*  restrict x_cast   = x0;
	const double*  restrict y_cast   = y0;
	      double*  restrict rho_cast = rho0;

	      dim_t             n_pre;
	      dim_t             n_run;
	      dim_t             n_left;

	      double            rho1;
	      double            x1c, y1c;

	      v2df_t            rho1v;
	      v2df_t            x1v, y1v;

	      bool              use_ref = FALSE;

	// If the vector lengths are zero, set rho to zero and return.
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(d,set0s)( *rho_cast );
		return;
	}

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
			n_pre = 1;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		dotv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_DOTV_KER, cntx );

		f
		(
		  conjx,
		  conjy,
		  n,
		  x0, incx,
		  y0, incy,
		  rho0,
		  cntx
		);
		return;
	}

	n_run       = ( n - n_pre ) / 2;
	n_left      = ( n - n_pre ) % 2;

	const double* restrict x1 = x_cast;
	const double* restrict y1 = y_cast;

	PASTEMAC(d,set0s)( rho1 );

	if ( n_pre == 1 )
	{
		x1c = *x1;
		y1c = *y1;

		rho1 += x1c * y1c;

		x1 += incx;
		y1 += incy;
	}

	rho1v.v = _mm_setzero_pd();

	for ( dim_t i = 0; i < n_run; ++i )
	{
		x1v.v = _mm_load_pd( ( double* )x1 );
		y1v.v = _mm_load_pd( ( double* )y1 );

		rho1v.v += x1v.v * y1v.v;

		//x1 += 2*incx;
		//y1 += 2*incy;
		x1 += 2;
		y1 += 2;
	}

	rho1 += rho1v.d[0] + rho1v.d[1];

	if ( n_left > 0 )
	{
		for ( dim_t i = 0; i < n_left; ++i )
		{
			x1c = *x1;
			y1c = *y1;

			rho1 += x1c * y1c;

			x1 += incx;
			y1 += incy;
		}
	}

	PASTEMAC(d,copys)( rho1, *rho_cast );
}
