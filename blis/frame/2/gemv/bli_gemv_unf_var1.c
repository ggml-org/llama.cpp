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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  A1; \
	ctype*  x1; \
	ctype*  y1; \
	dim_t   i; \
	dim_t   b_fuse, f; \
	dim_t   n_elem, n_iter; \
	inc_t   rs_at, cs_at; \
	conj_t  conja; \
\
	bli_set_dims_incs_with_trans( transa, \
	                              m, n, rs_a, cs_a, \
	                              &n_iter, &n_elem, &rs_at, &cs_at ); \
\
	conja = bli_extract_conj( transa ); \
\
	/* Query the context for the kernel function pointer and fusing factor. */ \
	dotxf_ker_ft kfp_df = bli_cntx_get_ukr_dt( dt, BLIS_DOTXF_KER, cntx ); \
	b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx ); \
\
	for ( i = 0; i < n_iter; i += f ) \
	{ \
		f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse ); \
\
		A1 = a + (i  )*rs_at + (0  )*cs_at; \
		x1 = x + (0  )*incy; \
		y1 = y + (i  )*incy; \
\
		/* y1 = beta * y1 + alpha * A1 * x; */ \
		kfp_df \
		( \
		  conja, \
		  conjx, \
		  n_elem, \
		  f, \
		  alpha, \
		  A1,   cs_at, rs_at, \
		  x1,   incx, \
		  beta, \
		  y1,   incy, \
		  cntx  \
		); \
\
	} \
}

INSERT_GENTFUNC_BASIC( gemv_unf_var1 )

