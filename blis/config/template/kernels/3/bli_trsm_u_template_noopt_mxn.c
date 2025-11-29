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


void bli_ztrsm_u_template_noopt
     (
       dcomplex*  restrict a11,
       dcomplex*  restrict b11,
       dcomplex*  restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t*          data,
       cntx_t*             cntx
     )
{
/*
  Template trsm_u micro-kernel implementation

  This function contains a template implementation for a double-precision
  complex trsm micro-kernel, coded in C, which can serve as the starting point
  for one to write an optimized micro-kernel on an arbitrary architecture.
  (We show a template implementation for only double-precision complex because
  the templates for the other three floating-point types would be nearly
  identical.)

  This micro-kernel performs the following operation:

    C11 := inv(A11) * B11

  where A11 is MR x MR and upper triangular, B11 is MR x NR, and C11 is
  MR x NR.

  For more info, please refer to the BLIS website's wiki on kernels:

    https://github.com/flame/blis/wiki/KernelsHowTo

  and/or contact the blis-devel mailing list.

  -FGVZ
*/
	const dim_t        mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t        nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

	const inc_t        packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx );
	const inc_t        packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	const dim_t        m      = mr;
	const dim_t        n      = nr;

	const inc_t        rs_a   = 1;
	const inc_t        cs_a   = packmr;

	const inc_t        rs_b   = packnr;
	const inc_t        cs_b   = 1;

	dim_t              iter, i, j, l;
	dim_t              n_behind;

	dcomplex* restrict alpha11;
	dcomplex* restrict a12t;
	dcomplex* restrict alpha12;
	dcomplex* restrict X2;
	dcomplex* restrict x1;
	dcomplex* restrict x21;
	dcomplex* restrict chi21;
	dcomplex* restrict chi11;
	dcomplex* restrict gamma11;
	dcomplex           rho11;

	for ( iter = 0; iter < m; ++iter )
	{
		i        = m - iter - 1;
		n_behind = iter;
		alpha11  = a11 + (i  )*rs_a + (i  )*cs_a;
		a12t     = a11 + (i  )*rs_a + (i+1)*cs_a;
		x1       = b11 + (i  )*rs_b + (0  )*cs_b;
		X2       = b11 + (i+1)*rs_b + (0  )*cs_b;

		/* x1 = x1 - a12t * X2; */
		/* x1 = x1 / alpha11; */
		for ( j = 0; j < n; ++j )
		{
			chi11   = x1  + (0  )*rs_b + (j  )*cs_b;
			x21     = X2  + (0  )*rs_b + (j  )*cs_b;
			gamma11 = c11 + (i  )*rs_c + (j  )*cs_c;

			/* chi11 = chi11 - a12t * x21; */
			bli_tset0s( z, rho11 );
			for ( l = 0; l < n_behind; ++l )
			{
				alpha12 = a12t + (l  )*cs_a;
				chi21   = x21  + (l  )*rs_b;

				bli_taxpys( z,z,z,z, *alpha12, *chi21, rho11 );
			}
			bli_tsubs( z,z,z, rho11, *chi11 );

			/* chi11 = chi11 / alpha11; */
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */
			bli_tscals( z,z,z, *alpha11, *chi11 );

			/* Output final result to matrix C. */
			bli_tcopys( z,z, *chi11, *gamma11 );
		}
	}
}

