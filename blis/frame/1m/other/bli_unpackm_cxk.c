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
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       conj_t  conjp, \
       pack_t  schema, \
       dim_t   panel_dim, \
       dim_t   panel_len, \
       ctype*  kappa, \
       ctype*  p,             inc_t ldp, \
       ctype*  a, inc_t inca, inc_t lda, \
       cntx_t* cntx  \
     ) \
{ \
	num_t dt     = PASTEMAC(ch,type); \
	ukr_t ker_id = bli_is_col_packed( schema ) ? BLIS_UNPACKM_NRXK_KER \
	                                           : BLIS_UNPACKM_MRXK_KER; \
\
	PASTECH2(ch,opname,_ker_ft) f; \
\
	/* Query the context for the unpackm kernel corresponding to the current
	   panel dimension, or kernel id. If the id is invalid, the function will
	   return NULL. */ \
	f = bli_cntx_get_ukr_dt( dt, ker_id, cntx ); \
\
	/* If there exists a kernel implementation for the micro-panel dimension
	   provided, we invoke the implementation. Otherwise, we use scal2m. */ \
	if ( f != NULL ) \
	{ \
		f \
		( \
		  conjp, \
		  panel_len, \
		  kappa, \
		  p,       ldp, \
		  a, inca, lda, \
		  cntx  \
		); \
	} \
	else \
	{ \
		trans_t transp = ( trans_t )conjp; \
\
		/* Treat the micro-panel as panel_dim x panel_len and column-stored
		   (unit row stride). */ \
		PASTEMAC2(ch,scal2m,BLIS_TAPI_EX_SUF) \
		( \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  transp, \
		  panel_dim, \
		  panel_len, \
		  kappa, \
		  p, 1,    ldp, \
		  a, inca, lda, \
		  cntx, \
		  NULL  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC0( unpackm_cxk )

