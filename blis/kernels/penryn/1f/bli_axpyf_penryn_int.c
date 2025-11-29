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


void bli_daxpyf_penryn_int
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha,
       const void*   a, inc_t inca, inc_t lda,
       const void*   x, inc_t incx,
             void*   y, inc_t incy,
       const cntx_t* cntx
     )
{
	const double*  restrict alpha_cast = alpha;
	const double*  restrict a_cast     = a;
	const double*  restrict x_cast     = x;
	      double*  restrict y_cast     = y;

	const dim_t             n_elem_per_reg = 2;
	const dim_t             n_iter_unroll  = 2;

	      dim_t             m_pre;
	      dim_t             m_run;
	      dim_t             m_left;

          double            a0c, a1c, a2c, a3c;
          double            chi0, chi1, chi2, chi3;

	      v2df_t            a00v, a01v, a02v, a03v, y0v;
	      v2df_t            a10v, a11v, a12v, a13v, y1v;
	      v2df_t            chi0v, chi1v, chi2v, chi3v;

	      bool              use_ref = FALSE;


	if ( bli_zero_dim2( m, b_n ) ) return;

	m_pre = 0;

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( b_n < bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_AF, cntx ) )
	{
		use_ref = TRUE;
	}
	else if ( inca != 1 || incx != 1 || incy != 1 ||
	          bli_is_unaligned_to( ( siz_t )(lda*sizeof(double)), 16 ) )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( ( siz_t )a, 16 ) ||
	          bli_is_unaligned_to( ( siz_t )y, 16 ) )
	{
		use_ref = TRUE;

		if ( bli_is_unaligned_to( ( siz_t )a, 16 ) &&
		     bli_is_unaligned_to( ( siz_t )y, 16 ) )
		{
			use_ref = FALSE;
			m_pre   = 1;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		#if 0
		axpyf_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYF_KER, cntx );

		f
		(
		  conja,
		  conjx,
		  m,
		  b_n,
		  alpha_cast,
		  a_cast, inca, lda,
		  x_cast, incx,
		  y_cast, incy,
		  cntx
		);
		#endif
		bli_abort();
		return;
	}


	m_run       = ( m - m_pre ) / ( n_elem_per_reg * n_iter_unroll );
	m_left      = ( m - m_pre ) % ( n_elem_per_reg * n_iter_unroll );

	const double* restrict a0   = a_cast + 0*lda;
	const double* restrict a1   = a_cast + 1*lda;
	const double* restrict a2   = a_cast + 2*lda;
	const double* restrict a3   = a_cast + 3*lda;
	      double* restrict y0   = y_cast;

	chi0 = *(x_cast + 0*incx);
	chi1 = *(x_cast + 1*incx);
	chi2 = *(x_cast + 2*incx);
	chi3 = *(x_cast + 3*incx);

	PASTEMAC(d,d,scals)( *alpha_cast, chi0 );
	PASTEMAC(d,d,scals)( *alpha_cast, chi1 );
	PASTEMAC(d,d,scals)( *alpha_cast, chi2 );
	PASTEMAC(d,d,scals)( *alpha_cast, chi3 );

	if ( m_pre == 1 )
	{
		a0c = *a0;
		a1c = *a1;
		a2c = *a2;
		a3c = *a3;

		*y0 += chi0 * a0c +
		       chi1 * a1c +
		       chi2 * a2c +
		       chi3 * a3c;

		a0 += inca;
		a1 += inca;
		a2 += inca;
		a3 += inca;
		y0 += incy;
	}

	chi0v.v = _mm_loaddup_pd( ( double* )&chi0 );
	chi1v.v = _mm_loaddup_pd( ( double* )&chi1 );
	chi2v.v = _mm_loaddup_pd( ( double* )&chi2 );
	chi3v.v = _mm_loaddup_pd( ( double* )&chi3 );

	for ( dim_t i = 0; i < m_run; ++i )
	{
		y0v.v = _mm_load_pd( ( double* )(y0 + 0*n_elem_per_reg) );

		a00v.v = _mm_load_pd( ( double* )(a0 + 0*n_elem_per_reg) );
		a01v.v = _mm_load_pd( ( double* )(a1 + 0*n_elem_per_reg) );

		y0v.v += chi0v.v * a00v.v;
		y0v.v += chi1v.v * a01v.v;

		a02v.v = _mm_load_pd( ( double* )(a2 + 0*n_elem_per_reg) );
		a03v.v = _mm_load_pd( ( double* )(a3 + 0*n_elem_per_reg) );

		y0v.v += chi2v.v * a02v.v;
		y0v.v += chi3v.v * a03v.v;

		_mm_store_pd( ( double* )(y0 + 0*n_elem_per_reg), y0v.v );


		y1v.v = _mm_load_pd( ( double* )(y0 + 1*n_elem_per_reg) );

		a10v.v = _mm_load_pd( ( double* )(a0 + 1*n_elem_per_reg) );
		a11v.v = _mm_load_pd( ( double* )(a1 + 1*n_elem_per_reg) );

		y1v.v += chi0v.v * a10v.v;
		y1v.v += chi1v.v * a11v.v;

		a12v.v = _mm_load_pd( ( double* )(a2 + 1*n_elem_per_reg) );
		a13v.v = _mm_load_pd( ( double* )(a3 + 1*n_elem_per_reg) );

		y1v.v += chi2v.v * a12v.v;
		y1v.v += chi3v.v * a13v.v;

		_mm_store_pd( ( double* )(y0 + 1*n_elem_per_reg), y1v.v );


		a0 += n_elem_per_reg * n_iter_unroll;
		a1 += n_elem_per_reg * n_iter_unroll;
		a2 += n_elem_per_reg * n_iter_unroll;
		a3 += n_elem_per_reg * n_iter_unroll;
		y0 += n_elem_per_reg * n_iter_unroll;
	}

	if ( m_left > 0 )
	{
		for ( dim_t i = 0; i < m_left; ++i )
		{
			a0c = *a0;
			a1c = *a1;
			a2c = *a2;
			a3c = *a3;

			*y0 += chi0 * a0c +
			       chi1 * a1c +
			       chi2 * a2c +
			       chi3 * a3c;

			a0 += inca;
			a1 += inca;
			a2 += inca;
			a3 += inca;
			y0 += incy;
		}
	}
}

