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

#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       conj_t  conja, \
       pack_t  schema, \
       dim_t   panel_dim, \
       dim_t   panel_dim_max, \
       dim_t   panel_len, \
       dim_t   panel_len_max, \
       ctype*  kappa, \
       ctype*  a, inc_t inca, inc_t lda, \
       ctype*  p,             inc_t ldp, \
       cntx_t* cntx  \
     ) \
{ \
	/* Note that we use panel_dim_max, not panel_dim, to query the packm
	   kernel function pointer. This means that we always use the same
	   kernel, even for edge cases. */ \
	num_t dt     = PASTEMAC(ch,type); \
	ukr_t ker_id = bli_is_col_packed( schema ) ? BLIS_PACKM_NRXK_1ER_KER \
	                                           : BLIS_PACKM_MRXK_1ER_KER; \
\
	PASTECH2(ch,opname,_ker_ft) f; \
\
	/* Query the context for the packm kernel corresponding to the current
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
		  conja, \
		  schema, \
		  panel_dim, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  a, inca, lda, \
		  p,       ldp, \
		  cntx  \
		); \
	} \
	else \
	{ \
		/* Treat the micro-panel as panel_dim x panel_len and column-stored
		   (unit row stride). */ \
\
		PASTEMAC(ch,scal21ms_mxn) \
		( \
		  schema, \
		  conja, \
		  panel_dim, \
		  panel_len, \
		  kappa, \
		  a, inca, lda, \
		  p, 1,    ldp, ldp  \
		); \
\
		/* If panel_dim < panel_dim_max, then we zero those unused rows. */ \
		if ( panel_dim < panel_dim_max ) \
		{ \
			ctype* restrict zero   = PASTEMAC(ch,0); \
			const dim_t     offm   = panel_dim; \
			const dim_t     offn   = 0; \
			const dim_t     m_edge = panel_dim_max - panel_dim; \
			const dim_t     n_edge = panel_len_max; \
\
			PASTEMAC(ch,set1ms_mxn) \
			( \
			  schema, \
			  offm, \
			  offn, \
			  m_edge, \
			  n_edge, \
			  zero, \
			  p, 1, ldp, ldp  \
			); \
		} \
\
		/* If panel_len < panel_len_max, then we zero those unused columns. */ \
		if ( panel_len < panel_len_max ) \
		{ \
			ctype* restrict zero   = PASTEMAC(ch,0); \
			const dim_t     offm   = 0; \
			const dim_t     offn   = panel_len; \
			const dim_t     m_edge = panel_dim_max; \
			const dim_t     n_edge = panel_len_max - panel_len; \
\
			PASTEMAC(ch,set1ms_mxn) \
			( \
			  schema, \
			  offm, \
			  offn, \
			  m_edge, \
			  n_edge, \
			  zero, \
			  p, 1, ldp, ldp  \
			); \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_cxk_1er )

