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


void bli_ddotaxpyv_penryn_int
     (
             conj_t  conjxt,
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const void*   alpha,
       const void*   x, inc_t incx,
       const void*   y, inc_t incy,
             void*   rho,
             void*   z, inc_t incz,
       const cntx_t* cntx
     )
{
	const double*  restrict alpha_cast = alpha;
	const double*  restrict x_cast     = x;
	const double*  restrict y_cast     = y;
	      double*  restrict rho_cast   = rho;
	      double*  restrict z_cast     = z;

	      dim_t             n_pre;
	      dim_t             n_run;
	      dim_t             n_left;

	      double            alpha1c, chi1c, psi1c, rho1c;
	      dim_t             i;
	      //inc_t             stepx, stepy, stepz;

	      v2df_t            alphav, rhov;
	      v2df_t            x1v, y1v, z1v;

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
		dotaxpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_DOTAXPYV_KER, cntx );

		f
		(
		  conjxt,
		  conjx,
		  conjy,
		  n,
		  alpha,
		  x, incx,
		  y, incy,
		  rho,
		  z, incz,
		  cntx
		);
		#endif
		bli_abort();
		return;
	}


	n_run       = ( n - n_pre ) / ( 2 * 1 );
	n_left      = ( n - n_pre ) % ( 2 * 1 );

	//stepx       = 2 * incx;
	//stepy       = 2 * incy;
	//stepz       = 2 * incz;

	PASTEMAC(d,set0s)( rho1c );

	alpha1c = *alpha_cast;

	const double* restrict chi1  = x_cast;
	const double* restrict psi1  = y_cast;
	      double* restrict zeta1 = z_cast;

	if ( n_pre == 1 )
	{
		chi1c  = *chi1;
		psi1c  = *psi1;

		rho1c  += chi1c * psi1c;
		*zeta1 += alpha1c * chi1c;

		chi1  += incx;
		psi1  += incy;
		zeta1 += incz;
	}

	rhov.v = _mm_setzero_pd();

	alphav.v = _mm_loaddup_pd( ( double* )alpha_cast );

	for ( i = 0; i < n_run; ++i )
	{
		x1v.v = _mm_load_pd( ( double* )chi1 );
		y1v.v = _mm_load_pd( ( double* )psi1 );
		z1v.v = _mm_load_pd( ( double* )zeta1 );
		//y1v.v = _mm_setr_pd( *psi1,  *(psi1  + incy) );
		//z1v.v = _mm_setr_pd( *zeta1, *(zeta1 + incz) );

		rhov.v += x1v.v * y1v.v;
		z1v.v  += alphav.v * x1v.v;

		_mm_store_pd( ( double* )zeta1, z1v.v );

		//chi1  += stepx;
		//psi1  += stepy;
		//zeta1 += stepz;
		chi1  += 2;
		psi1  += 2;
		zeta1 += 2;
	}

	if ( n_left > 0 )
	{
		for ( i = 0; i < n_left; ++i )
		{
			chi1c  = *chi1;
			psi1c  = *psi1;

			rho1c += chi1c * psi1c;
			*zeta1 += alpha1c * chi1c;

			chi1  += incx;
			psi1  += incy;
			zeta1 += incz;
		}
	}

	rho1c += rhov.d[0] + rhov.d[1];

	*rho_cast = rho1c;
}

