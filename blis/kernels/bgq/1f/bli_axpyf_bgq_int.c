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

#include "blis.h"


void bli_daxpyf_bgq_int
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* alpha = alpha0;
	const double* a     = a0;
	const double* x     = x0;
	      double* y     = y0;

	const dim_t fusefac = 8;

    if ( bli_zero_dim2( m, b_n ) ) return;

	bool              use_ref = FALSE;
//    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", b_n, fusefac, inca, incx, incy, bli_is_unaligned_to( ( siz_t )a, 32 ), bli_is_unaligned_to( ( siz_t )y, 32));
	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( ( b_n < fusefac) || inca != 1 || incx != 1 || incy != 1 || bli_is_unaligned_to( ( siz_t )a, 32 ) || bli_is_unaligned_to( ( siz_t )y, 32 ) )
		use_ref = TRUE;
	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
//        printf("%d\t%d\t%d\t%d\t%d\t%d\n", fusefac, inca, incx, incy, bli_is_unaligned_to( ( siz_t )a, 32 ), bli_is_unaligned_to( ( siz_t )y, 32));
//        printf("DEFAULTING TO REFERENCE IMPLEMENTATION\n");
		#if 0
		axpyf_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYF_KER, cntx );

		f
		(
		  conja,
		  conjx,
		  m,
		  b_n,
		  alpha0,
		  a0, inca, lda,
		  x0, incx,
		  y0, incy,
		  cntx
		);
		#endif
		bli_abort();
		return;
	}

	dim_t m_run       =  m / 4;
	dim_t m_left      =  m % 4;

	const double* ap0   = a + 0*lda;
	const double* ap1   = a + 1*lda;
	const double* ap2   = a + 2*lda;
	const double* ap3   = a + 3*lda;
	const double* ap4   = a + 4*lda;
	const double* ap5   = a + 5*lda;
	const double* ap6   = a + 6*lda;
	const double* ap7   = a + 7*lda;
	      double* yp0   = y;

	double chi0 = *(x + 0*incx);
	double chi1 = *(x + 1*incx);
	double chi2 = *(x + 2*incx);
	double chi3 = *(x + 3*incx);
	double chi4 = *(x + 4*incx);
	double chi5 = *(x + 5*incx);
	double chi6 = *(x + 6*incx);
	double chi7 = *(x + 7*incx);

	PASTEMAC(d,d,scals)( *alpha, chi0 );
	PASTEMAC(d,d,scals)( *alpha, chi1 );
	PASTEMAC(d,d,scals)( *alpha, chi2 );
	PASTEMAC(d,d,scals)( *alpha, chi3 );
	PASTEMAC(d,d,scals)( *alpha, chi4 );
	PASTEMAC(d,d,scals)( *alpha, chi5 );
	PASTEMAC(d,d,scals)( *alpha, chi6 );
	PASTEMAC(d,d,scals)( *alpha, chi7 );

	vector4double   a0v, a1v, a2v, a3v, a4v, a5v, a6v, a7v;
    vector4double   yv;
	vector4double   chi0v, chi1v, chi2v, chi3v, chi4v, chi5v, chi6v, chi7v;
	chi0v = vec_splats( chi0 );
	chi1v = vec_splats( chi1 );
	chi2v = vec_splats( chi2 );
	chi3v = vec_splats( chi3 );
	chi4v = vec_splats( chi4 );
	chi5v = vec_splats( chi5 );
	chi6v = vec_splats( chi6 );
	chi7v = vec_splats( chi7 );

    for ( dim_t i = 0; i < m_run; i += 1 )
	{
		yv  = vec_lda( 0 * sizeof(double), &yp0[i*4]);

		a0v = vec_lda( 0 * sizeof(double), &ap0[i*4]);
		a1v = vec_lda( 0 * sizeof(double), &ap1[i*4]);
		a2v = vec_lda( 0 * sizeof(double), &ap2[i*4]);
		a3v = vec_lda( 0 * sizeof(double), &ap3[i*4]);
		a4v = vec_lda( 0 * sizeof(double), &ap4[i*4]);
		a5v = vec_lda( 0 * sizeof(double), &ap5[i*4]);
		a6v = vec_lda( 0 * sizeof(double), &ap6[i*4]);
		a7v = vec_lda( 0 * sizeof(double), &ap7[i*4]);

        yv = vec_madd( chi0v, a0v, yv );
        yv = vec_madd( chi1v, a1v, yv );
        yv = vec_madd( chi2v, a2v, yv );
        yv = vec_madd( chi3v, a3v, yv );
        yv = vec_madd( chi4v, a4v, yv );
        yv = vec_madd( chi5v, a5v, yv );
        yv = vec_madd( chi6v, a6v, yv );
        yv = vec_madd( chi7v, a7v, yv );

        vec_sta( yv, 0 * sizeof(double), &yp0[i*4]);
	}

    for ( dim_t i = 0; i < m_left; ++i )
    {
        yp0[4*m_run + i] += chi0 * ap0[4*m_run + i]
                         +  chi1 * ap1[4*m_run + i]
                         +  chi2 * ap2[4*m_run + i]
                         +  chi3 * ap3[4*m_run + i]
                         +  chi4 * ap4[4*m_run + i]
                         +  chi5 * ap5[4*m_run + i]
                         +  chi6 * ap6[4*m_run + i]
                         +  chi7 * ap7[4*m_run + i];
    }

}

