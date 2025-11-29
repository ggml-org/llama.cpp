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

#include <omp.h>

void bli_ddotv_bgq_int
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
	const double* x   = x0;
	const double* y   = y0;
	const double* rho = rho0;

	bool use_ref = FALSE;

	// If the vector lengths are zero, set rho to zero and return.
	if ( bli_zero_dim1( n ) ) {
		PASTEMAC(d,set0s)( *rho );
		return;
	}
	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( incx != 1 || incy != 1 || bli_is_unaligned_to( ( siz_t )x, 32 ) || bli_is_unaligned_to( ( siz_t )y, 32 ) )
		use_ref = TRUE;
	// Call the reference implementation if needed.
	if ( use_ref ) {
		BLIS_DDOTV_KERNEL_REF( conjx, conjy, n, x, incx, y, incy, rho, cntx );
		return;
	}

	dim_t n_run       = n / 4;
	dim_t n_left      = n % 4;

    double rhos = 0.0;
    #pragma omp parallel reduction(+:rhos)
    {
        dim_t n_threads;
        dim_t t_id = omp_get_thread_num();
        n_threads = omp_get_num_threads();
        vector4double rhov = vec_splats( 0.0 );
        vector4double xv, yv;

        for ( dim_t i = t_id; i < n_run; i += n_threads )
        {
            xv = vec_lda( 0 * sizeof(double), &x[i*4] );
            yv = vec_lda( 0 * sizeof(double), &y[i*4] );

            rhov = vec_madd( xv, yv, rhov );
        }

        rhos += vec_extract( rhov, 0 );
        rhos += vec_extract( rhov, 1 );
        rhos += vec_extract( rhov, 2 );
        rhos += vec_extract( rhov, 3 );
    }
    for ( dim_t i = 0; i < n_left; i++ )
    {
        rhos += x[4*n_run + i] * y[4*n_run + i];
    }

    *rho = rhos;
}

