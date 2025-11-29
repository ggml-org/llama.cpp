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
void PASTECH2(bao_,ch,opname) \
     ( \
       conj_t  conja, \
       pack_t  schema, \
       dim_t   panel_dim, \
       dim_t   panel_dim_max, \
       dim_t   panel_len, \
       dim_t   panel_len_max, \
       ctype*  kappa, \
       ctype*  d, inc_t incd, \
       ctype*  a, inc_t inca, inc_t lda, \
       ctype*  p,             inc_t ldp, \
       cntx_t* cntx  \
     ) \
{ \
	/* Note that we use panel_dim_max, not panel_dim, to query the packm
	   kernel function pointer. This means that we always use the same
	   kernel, even for edge cases. */ \
	num_t dt     = PASTEMAC(ch,type); \
	ukr_t ker_id = bli_is_col_packed( schema ) ? BLIS_PACKM_NRXK_KER : BLIS_PACKM_MRXK_KER; \
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
	/* NOTE: We've disabled calling packm micro-kernels from the context for
	   this implementation. To re-enable, change FALSE to TRUE in the
	   conditional below. */ \
	if ( f != NULL && FALSE ) \
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
		/* NOTE: We assume here that kappa = 1 and therefore ignore it. If
		   we're wrong, this will get someone's attention. */ \
		if ( !PASTEMAC(ch,eq1)( *kappa ) ) \
			bli_abort(); \
\
		if ( d == NULL ) \
		{ \
			/* Perform the packing, taking conja into account. */ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t l = 0; l < panel_len; ++l ) \
				{ \
					for ( dim_t i = 0; i < panel_dim; ++i ) \
					{ \
						ctype* ali = a + (l  )*lda + (i  )*inca; \
						ctype* pli = p + (l  )*ldp + (i  )*1; \
\
						PASTEMAC(ch,copyjs)( *ali, *pli ); \
					} \
				} \
			} \
			else \
			{ \
				for ( dim_t l = 0; l < panel_len; ++l ) \
				{ \
					for ( dim_t i = 0; i < panel_dim; ++i ) \
					{ \
						ctype* ali = a + (l  )*lda + (i  )*inca; \
						ctype* pli = p + (l  )*ldp + (i  )*1; \
\
						PASTEMAC(ch,copys)( *ali, *pli ); \
					} \
				} \
			} \
		} \
		else /* if ( d != NULL ) */ \
		{ \
			/* Perform the packing, taking conja into account. */ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t l = 0; l < panel_len; ++l ) \
				{ \
					for ( dim_t i = 0; i < panel_dim; ++i ) \
					{ \
						ctype* ali = a + (l  )*lda + (i  )*inca; \
						ctype* dl  = d + (l  )*incd; \
						ctype* pli = p + (l  )*ldp + (i  )*1; \
\
						/* Note that ali must be the second operand here since
						   that is what is conjugated by scal2js. */ \
						PASTEMAC(ch,scal2js)( *dl, *ali, *pli ); \
					} \
				} \
			} \
			else \
			{ \
				for ( dim_t l = 0; l < panel_len; ++l ) \
				{ \
					for ( dim_t i = 0; i < panel_dim; ++i ) \
					{ \
						ctype* ali = a + (l  )*lda + (i  )*inca; \
						ctype* dl  = d + (l  )*incd; \
						ctype* pli = p + (l  )*ldp + (i  )*1; \
\
						PASTEMAC(ch,scal2s)( *ali, *dl, *pli ); \
					} \
				} \
			} \
		} \
\
		/* If panel_dim < panel_dim_max, then we zero those unused rows. */ \
		if ( panel_dim < panel_dim_max ) \
		{ \
			const dim_t     i      = panel_dim; \
			const dim_t     m_edge = panel_dim_max - panel_dim; \
			const dim_t     n_edge = panel_len_max; \
			ctype* restrict p_edge = p + (i  )*1; \
\
			PASTEMAC(ch,set0s_mxn) \
			( \
			  m_edge, \
			  n_edge, \
			  p_edge, 1, ldp  \
			); \
		} \
\
		/* If panel_len < panel_len_max, then we zero those unused columns. */ \
		if ( panel_len < panel_len_max ) \
		{ \
			const dim_t     j      = panel_len; \
			const dim_t     m_edge = panel_dim_max; \
			const dim_t     n_edge = panel_len_max - panel_len; \
			ctype* restrict p_edge = p + (j  )*ldp; \
\
			PASTEMAC(ch,set0s_mxn) \
			( \
			  m_edge, \
			  n_edge, \
			  p_edge, 1, ldp  \
			); \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC0( packm_cxk )
GENTFUNC( float,    s, packm_cxk )
GENTFUNC( double,   d, packm_cxk )
GENTFUNC( scomplex, c, packm_cxk )
GENTFUNC( dcomplex, z, packm_cxk )

