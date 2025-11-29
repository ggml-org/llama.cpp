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

void bli_daxpyv_bgq_int
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* alpha = alpha0;
	const double* x     = x0;
	const double* y     = y0;

	if ( bli_zero_dim1( n ) ) return;

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	bool   use_ref = FALSE;
	if ( incx != 1 || incy != 1 || bli_is_unaligned_to( ( siz_t )x, 32 ) || bli_is_unaligned_to( ( siz_t )y, 32 ) ) {
		use_ref = TRUE;
	}
	// Call the reference implementation if needed.
	if ( use_ref == TRUE ) {
		BLIS_DAXPYV_KERNEL_REF( conjx, n, alpha, x, incx, y, incy, cntx );
		return;
	}

	dim_t n_run       = n / 4;
    dim_t n_left      = n % 4;

    vector4double xv, yv, zv;
    vector4double alphav = vec_lds( 0 * sizeof(double), (double*)alpha );

    #pragma omp parallel for
	for ( dim_t i = 0; i < n_run; i++ )
	{
        xv = vec_lda( 0 * sizeof(double), &x[i*4] );
        yv = vec_lda( 0 * sizeof(double), &y[i*4] );
        zv = vec_madd( alphav, xv, yv );
        vec_sta( zv, 0 * sizeof(double), &y[i*4] );
	}
    for ( dim_t i = 0; i < n_left; i++ )
    {
        y[4*n_run + i] += *alpha * x[4*n_run + i];
    }
}
