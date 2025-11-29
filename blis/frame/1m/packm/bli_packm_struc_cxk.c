/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021-2025, Southern Methodist University

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

#undef  GENTFUNC2RO
#define GENTFUNC2RO( ctypec_r, ctype_c, ctypep_r, ctypep, chc_r, chc, chp_r, chp, varname ) \
GENTFUNC2RO_( ctypec_r, ctypec_r, ctypep_r, ctypep_r, chc_r, chc_r, chp_r, chp_r, varname ) \
GENTFUNC2RO_( ctypec_r, ctypec,   ctypep_r, ctypep,   chc_r, chc,   chp_r, chp,   varname )

#undef  GENTFUNC2RO_
#define GENTFUNC2RO_( ctypec_r, ctype_c, ctypep_r, ctypep, chc_r, chc, chp_r, chp, varname ) \
\
void PASTEMAC(chc,chp,varname) \
     ( \
             struc_t strucc, \
             diag_t  diagc, \
             uplo_t  uploc, \
             conj_t  conjc, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   panel_dim, \
             dim_t   panel_len, \
             dim_t   panel_dim_max, \
             dim_t   panel_len_max, \
             dim_t   panel_dim_off, \
             dim_t   panel_len_off, \
             dim_t   panel_bcast, \
       const void*   kappa, \
       const void*   c, inc_t incc, inc_t ldc, \
             void*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx \
     ) \
{ \
	num_t dt_c          = PASTEMAC(chc,type); \
	num_t dt_p          = PASTEMAC(chp,type); \
	num_t dt_p0         = dt_p; \
\
	/* Always do pointer arithmetic in the real domain so that we
	   can cleanly handle the real-only packing case. */ \
	inc_t incc_r        = incc; \
	inc_t ldc_r         = ldc; \
	inc_t ldp_r         = ldp; \
\
	if ( bli_is_complex( dt_c ) ) \
	{ \
		incc_r *= 2; \
		ldc_r *= 2; \
		ldp_r *= 2; \
	} \
\
	dim_t panel_len_pad = panel_len_max - panel_len; \
\
	ukr_t cxk_ker_id    = BLIS_PACKM_KER; \
	ukr_t cxc_ker_id    = BLIS_PACKM_DIAG_KER; \
\
	if ( bli_is_1m_packed( schema ) ) \
	{ \
		cxk_ker_id = BLIS_PACKM_1ER_KER; \
		cxc_ker_id = BLIS_PACKM_DIAG_1ER_KER; \
	} \
	else if ( bli_is_ro_packed( schema ) ) \
	{ \
		ctypep_r kappa_r, kappa_i; \
		( void )kappa_r; \
		bli_tgets( chp,chp, *( ctypep* )kappa, kappa_r, kappa_i ); \
		if ( bli_teq0s( chp_r, kappa_i ) ) \
		{ \
			/* Treat the matrix as real with doubled strides. */ \
			dt_c = bli_dt_proj_to_real( dt_c ); \
			dt_p = bli_dt_proj_to_real( dt_p ); \
			incc *= 2; \
			ldc *= 2; \
			schema = BLIS_PACKED_PANELS; \
		} \
		else \
		{ \
			cxk_ker_id = BLIS_PACKM_RO_KER; \
			cxc_ker_id = BLIS_PACKM_DIAG_RO_KER; \
		} \
\
		/* Make sure that P is treated as a real matrix. */ \
		ldp_r /= 2; \
		dt_p0 = bli_dt_proj_to_real( dt_p ); \
	} \
\
	const void*           zero   = bli_obj_buffer_for_const( dt_p0, &BLIS_ZERO ); \
	setv_ker_ft           f_setv = bli_cntx_get_ukr_dt( dt_p0, BLIS_SETV_KER, cntx ); \
	packm_cxk_ker_ft      f_cxk  = bli_cntx_get_ukr2_dt( dt_c, dt_p, cxk_ker_id, cntx ); \
	packm_cxc_diag_ker_ft f_cxc  = bli_cntx_get_ukr2_dt( dt_c, dt_p, cxc_ker_id, cntx ); \
\
	/* For general matrices, pack and return early */ \
	if ( bli_is_general( strucc ) ) \
	{ \
		f_cxk \
		( \
		  conjc, \
		  schema, \
		  panel_dim, \
		  panel_dim_max, \
		  panel_bcast, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  c, incc, ldc, \
		  p,       ldp, \
		  params, \
		  cntx \
		); \
		return; \
	} \
\
	/* Sanity check. Diagonals should not intersect the short end of
	   a micro-panel. If they do, then somehow the constraints on
	   cache blocksizes being a whole multiple of the register
	   blocksizes was somehow violated. */ \
	doff_t diagoffc = panel_dim_off - panel_len_off; \
	if ( (          -panel_dim < diagoffc && diagoffc <         0 ) || \
		 ( panel_len-panel_dim < diagoffc && diagoffc < panel_len ) ) \
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
	/* For triangular, symmetric, and hermitian matrices we need to consider
	   three parts. */ \
\
	/* Pack to p10. */ \
	if ( 0 < diagoffc ) \
	{ \
		dim_t     p10_len     = bli_min( diagoffc, panel_len ); \
		dim_t     p10_len_max = p10_len == panel_len ? panel_len_max : p10_len; \
		ctypep_r* p10         = ( ctypep_r* )p; \
		conj_t    conjc10     = conjc; \
		ctypec_r* c10         = ( ctypec_r* )c; \
		inc_t     incc10_r    = incc_r; \
		inc_t     ldc10_r     = ldc_r; \
		inc_t     incc10      = incc; \
		inc_t     ldc10       = ldc; \
\
		if ( bli_is_upper( uploc ) ) \
		{ \
			bli_reflect_to_stored_part( diagoffc, c10, incc10_r, ldc10_r ); \
			bli_swap_incs(&incc10, &ldc10); \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc10 ); \
		} \
\
		/* If we are referencing the unstored part of a triangular matrix,
		   explicitly store zeros */ \
		if ( bli_is_upper( uploc ) && bli_is_triangular( strucc ) ) \
		{ \
			f_setv \
			( \
			  BLIS_NO_CONJUGATE, \
			  ldp * p10_len_max, \
			  zero, \
			  p10, 1, \
			  cntx \
			); \
		} \
		else \
		{ \
			f_cxk \
			( \
			  conjc10, \
			  schema, \
			  panel_dim, \
			  panel_dim_max, \
			  panel_bcast, \
			  p10_len, \
			  p10_len_max, \
			  kappa, \
			  c10, incc10, ldc10, \
			  p10,         ldp, \
			  params, \
			  cntx \
			); \
		} \
	} \
\
	/* Pack to p11. */ \
	if ( 0 <= diagoffc && diagoffc + panel_dim <= panel_len ) \
	{ \
		dim_t     i           = diagoffc; \
		dim_t     p11_len_max = panel_dim + ( diagoffc + panel_dim == panel_len \
		                                   ? panel_len_pad : 0 ); \
		ctypep_r* p11         = ( ctypep_r* )p + i * ldp_r; \
		conj_t    conjc11     = conjc; \
		ctypec_r* c11         = ( ctypec_r* )c + i * ldc_r; \
		inc_t     incc11      = incc; \
		inc_t     ldc11       = ldc; \
\
		f_cxc \
		( \
		  strucc, \
		  diagc, \
		  uploc, \
		  conjc11, \
		  schema, \
		  invdiag, \
		  panel_dim, \
		  panel_dim_max, \
		  panel_bcast, \
		  p11_len_max, \
		  kappa, \
		  c11, incc11, ldc11, \
		  p11,         ldp, \
		  params, \
		  cntx \
		); \
	} \
\
	/* Pack to p12. */ \
	if ( diagoffc + panel_dim < panel_len ) \
	{ \
		dim_t     i           = bli_max( 0, diagoffc + panel_dim ); \
		dim_t     p12_len     = panel_len - i; \
		/* If we are packing p12, then it is always the last partial block
		   and so we should make sure to pad with zeros if necessary. */ \
		dim_t     p12_len_max = p12_len + panel_len_pad; \
		ctypep_r* p12         = ( ctypep_r* )p + i * ldp_r; \
		conj_t    conjc12     = conjc; \
		ctypec_r* c12         = ( ctypec_r* )c + i * ldc_r; \
		inc_t     incc12_r    = incc_r; \
		inc_t     ldc12_r     = ldc_r; \
		inc_t     incc12      = incc; \
		inc_t     ldc12       = ldc; \
\
		if ( bli_is_lower( uploc ) ) \
		{ \
			bli_reflect_to_stored_part( diagoffc - i, c12, incc12_r, ldc12_r ); \
			bli_swap_incs(&incc12, &ldc12); \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc12 ); \
		} \
\
		/* If we are referencing the unstored part of a triangular matrix,
		   explicitly store zeros */ \
		if ( bli_is_lower( uploc ) && bli_is_triangular( strucc ) ) \
		{ \
			f_setv \
			( \
			  BLIS_NO_CONJUGATE, \
			  ldp * p12_len_max, \
			  zero, \
			  p12, 1, \
			  cntx \
			); \
		} \
		else \
		{ \
			f_cxk \
			( \
			  conjc12, \
			  schema, \
			  panel_dim, \
			  panel_dim_max, \
			  panel_bcast, \
			  p12_len, \
			  p12_len_max, \
			  kappa, \
			  c12, incc12, ldc12, \
			  p12,         ldp, \
			  params, \
			  cntx \
			); \
		} \
	} \
}

INSERT_GENTFUNC2RO( packm_struc_cxk )
INSERT_GENTFUNC2RO_MIX_P( packm_struc_cxk )

