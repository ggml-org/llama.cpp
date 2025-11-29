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


void bli_zgemmtrsm_l_template_noopt
     (
       dim_t               k,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a10,
       dcomplex*  restrict a11,
       dcomplex*  restrict b01,
       dcomplex*  restrict b11,
       dcomplex*  restrict c11, inc_t rs_c, inc_t cs_c,
       auxinfo_t*          data,
       cntx_t*             cntx
     )
{
/*
  Template gemmtrsm_l micro-kernel implementation

  This function contains a template implementation for a double-precision
  complex micro-kernel that fuses a gemm with a trsm_l subproblem.

  This micro-kernel performs the following compound operation:

    B11 := alpha * B11 - A10 * B01    (gemm)
    B11 := inv(A11) * B11             (trsm)
    C11 := B11

  where A11 is MR x MR and lower triangular, A10 is MR x k, B01 is k x NR,
  B11 is MR x NR, and alpha is a scalar. Here, inv() denotes matrix
  inverse.

  For more info, please refer to the BLIS website's wiki on kernels:

    https://github.com/flame/blis/wiki/KernelsHowTo

  and/or contact the blis-devel mailing list.

  -FGVZ
*/
	const num_t        dt        = BLIS_DCOMPLEX;

	const inc_t        mr        = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const inc_t        nr        = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	const inc_t        packnr    = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx );

	const inc_t        rs_b      = packnr;
	const inc_t        cs_b      = 1;

	dcomplex* restrict minus_one = bli_zm1;

	/* b11 = alpha * b11 - a10 * b01; */
	bli_zgemm_template_noopt
	(
	  mr,
	  nr,
	  k,
	  minus_one,
	  a10,
	  b01,
	  alpha,
	  b11, rs_b, cs_b,
	  data
	);

	/* b11 = inv(a11) * b11;
	   c11 = b11; */
	bli_ztrsm_l_template_noopt
	(
	  a11,
	  b11,
	  c11, rs_c, cs_c,
	  data
	);
}


