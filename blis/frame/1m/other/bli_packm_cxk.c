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
	ukr_t ker_id = bli_is_col_packed( schema ) ? BLIS_PACKM_NRXK_KER \
	                                           : BLIS_PACKM_MRXK_KER; \
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
		/* Under normal circumstances, the packm kernel will copy over a
		   panel_dim x panel_len submatrix of A into P. However, the kernel
		   now handles zero-filling at edge cases, which typically consist of
		   the outer (panel_dim_max - panel_dim) rows or columns of the
		   micropanel. (Note that these rows/columns correspond to values
		   beyond the edge of matrix A.) The kernel intrinsically knows its
		   own panel_dim_max, since that corresponds to the packm micropanel's
		   normal width (corresponding to the gemm microkernel's register
		   blocksize (mr or nr). However, we *do* need to pass in panel_len_max
		   because the bottom-right edge case of trsm_lu will need all
		   elements above the extended diagonal and beyond (to the right of)
		   the bottom-right element to be initialized to zero so the trsm
		   portion of the computational kernel will operate with zeros for
		   those iterations.

		   For example, if trsm_lu is executed on an 10x10 triangular matrix,
		   and the gemmtrsm kernel uses MR = 6, the computation will begin
		   with the edge case, which is the bottom-right 4x4 upper triangular
		   matrix. Code in bli_packm_tri_cxk() will extend the diagonal as
		   identity into the remaining portion of the micropanel. But before
		   that happens, the packm kernel must have set the 0's added in
		   step (3) below.

		     packm kernel     packm kernel     packm kernel     packm_tri_cxk
		     step 1:          step 2:          step 3:          step 4:

		     x x x x . .      x x x x . .      x x x x 0 0      x x x x 0 0
		     ? x x x . .      ? x x x . .      ? x x x 0 0      ? x x x 0 0
		     ? ? x x . .  ->  ? ? x x . .  ->  ? ? x x 0 0  ->  ? ? x x 0 0
		     ? ? ? x . .      ? ? ? x . .      ? ? ? x 0 0      ? ? ? x 0 0
		     . . . . . .      0 0 0 0 0 0      0 0 0 0 0 0      0 0 0 0 1 0
		     . . . . . .      0 0 0 0 0 0      0 0 0 0 0 0      0 0 0 0 0 1

		     x  Copied from A; valid element.
		     ?  Copied from A, but value is unknown and unused.
		     .  Uninitialized.
		     0  Initialized to zero.
		     1  Initialized to one.

		     NOTE: In step 5 (not shown), bli_packm_tri_cxk() sets the ?'s
		     to zero. This is not needed to support trsm, but rather to
		     support trmm. (Both use the same packing format and code.)

		   In this case, panel_dim will be 4 because four rows of data are
		   copied from A, panel_len will be 4 because those four rows span
		   four columns of A, and panel_len_max will be 6 because there are a
		   total of 6 columns that can be written to in the packed micropanel,
		   2 of which lie beyond the values copied from A. */ \
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
		PASTEMAC2(ch,scal2m,BLIS_TAPI_EX_SUF) \
		( \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  ( trans_t )conja, \
		  panel_dim, \
		  panel_len, \
		  kappa, \
		  a, inca, lda, \
		  p, 1,    ldp, \
		  cntx, \
		  /* The rntm_t* can safely be NULL as long as it's not used by
		     scal2m_ex(). */ \
		  NULL  \
		); \
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

INSERT_GENTFUNC_BASIC0( packm_cxk )

