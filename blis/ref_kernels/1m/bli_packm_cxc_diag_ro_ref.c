/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Southern Methodist University

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


#define PACKM_SET_RO( chp_r, val, mnk ) \
do { \
	bli_tcopys( chp_r,chp_r, val, *(pi1_r + mnk*cdim_bcast + d + mnk*ldp) ); \
} while (0)


#define PACKM_SCAL_RO( ctypep_r, cha, chp, chp_r, mn, k, op ) \
do { \
	ctypep_r ka_r, ka_i; (void)ka_i; \
	PASTEMAC(t,op)( chp,cha,chp,chp, \
	                kappa_r, kappa_i, \
	                *(alpha1 + mn*inca2 + 0 + k*lda2), \
	                *(alpha1 + mn*inca2 + 1 + k*lda2), \
	                ka_r, ka_i ); \
	bli_tcopys( chp_r,chp_r, ka_r, *(pi1_r + mn*cdim_bcast + d + k*ldp) ); \
} while (0)


#define PACKM_DIAG_RO_BODY( ctypep_r, cha, chp, chp_r, mn_min, mn_max, inca2_lu, lda2_lu, op ) \
\
do \
{ \
	/* PACKM_SCAL_RO assumes inca2 and lda2 are the strides to use. */ \
	dim_t inca2 = inca2_lu; \
	dim_t lda2 = lda2_lu; \
	for ( dim_t k = 0; k < cdim; k++ ) \
	for ( dim_t mn = mn_min; mn < mn_max; mn++ ) \
	for ( dim_t d = 0; d < cdim_bcast; d++ ) \
		PACKM_SCAL_RO( ctypep_r, cha, chp, chp_r, mn, k, op ); \
} while(0)


#define PACKM_DIAG_BODY_RO_L( ctypep_r, cha, chp, chp_r, op ) \
	PACKM_DIAG_RO_BODY( ctypep_r, cha, chp, chp_r, k+1, cdim, inca_l2, lda_l2, op )

#define PACKM_DIAG_BODY_RO_U( ctypep_r, cha, chp, chp_r, op ) \
	PACKM_DIAG_RO_BODY( ctypep_r, cha, chp, chp_r, 0, k, inca_u2, lda_u2, op )


#undef  GENTFUNC2R
#define GENTFUNC2R( ctypea, ctypea_r, cha, cha_r, ctypep, ctypep_r, chp, chp_r, opname, arch, suf ) \
\
void PASTEMAC(cha,chp,opname,arch,suf) \
     ( \
             struc_t struca, \
             diag_t  diaga, \
             uplo_t  uploa, \
             conj_t  conja, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   cdim, \
             dim_t   cdim_max, \
             dim_t   cdim_bcast, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
             void*   p,             inc_t ldp, \
       const void*   params, \
       const cntx_t* cntx \
     ) \
{ \
	const inc_t inca2 = 2 * inca; \
	const inc_t lda2  = 2 * lda; \
\
	      ctypep_r           kappa_r = ( ( ctypep_r* )kappa )[0]; \
	      ctypep_r           kappa_i = ( ( ctypep_r* )kappa )[1]; \
	      ctypep_r           one     = *PASTEMAC(chp_r,1); \
	const ctypea_r* restrict alpha1  = ( const ctypea_r* )a; \
\
	/* start by zeroing out the whole block */ \
	bli_tset0s_mxn \
	( \
	  chp_r, \
	  cdim_max, \
	  n_max, \
	  ( ctypep_r* )p, 1, ldp  \
	); \
\
	ctypep_r* restrict pi1_r = ( ctypep_r* )p; \
\
	/* write the strictly lower part if it exists */ \
	if ( bli_is_lower( uploa ) || bli_is_herm_or_symm( struca ) ) \
	{ \
		dim_t  inca_l2 = inca2; \
		dim_t  lda_l2  = lda2; \
		conj_t conja_l = conja; \
\
		if ( bli_is_upper( uploa ) ) \
		{ \
			bli_swap_incs( &inca_l2, &lda_l2 ); \
			if ( bli_is_hermitian( struca ) ) \
			    bli_toggle_conj( &conja_l ); \
		} \
\
		if ( bli_is_conj( conja_l ) ) PACKM_DIAG_BODY_RO_L( ctypep_r, cha, chp, chp_r, scal2jris ); \
		else                          PACKM_DIAG_BODY_RO_L( ctypep_r, cha, chp, chp_r, scal2ris ); \
	} \
\
	/* write the strictly upper part if it exists */ \
	/* assume either symmetric, hermitian, or triangular */ \
	if ( bli_is_upper( uploa ) || bli_is_herm_or_symm( struca ) ) \
	{ \
		dim_t  inca_u2 = inca2; \
		dim_t  lda_u2  = lda2; \
		conj_t conja_u = conja; \
\
		if ( bli_is_lower( uploa ) ) \
		{ \
			bli_swap_incs( &inca_u2, &lda_u2 ); \
			if ( bli_is_hermitian( struca ) ) \
			    bli_toggle_conj( &conja_u ); \
		} \
\
		if ( bli_is_conj( conja_u ) ) PACKM_DIAG_BODY_RO_U( ctypep_r, cha, chp, chp_r, scal2jris ); \
		else                          PACKM_DIAG_BODY_RO_U( ctypep_r, cha, chp, chp_r, scal2ris ); \
	} \
\
	/* write the diagonal */ \
	if ( bli_is_unit_diag( diaga ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PACKM_SET_RO( chp_r, kappa_r, mnk ); \
	} \
	else if ( bli_is_hermitian( struca ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			bli_tscal2s( chp_r,cha_r,chp_r,chp_r, \
			             kappa_r, \
			             *(alpha1 + mnk*(inca2 + lda2)), \
			             *(pi1_r + mnk*(cdim_bcast + ldp) + d) ); \
	} \
	else if ( bli_is_conj( conja ) ) \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PACKM_SCAL_RO( ctypep_r, cha, chp, chp_r, mnk, mnk, scal2jris ); \
	} \
	else \
	{ \
		for ( dim_t mnk = 0; mnk < cdim; ++mnk ) \
		for ( dim_t d = 0; d < cdim_bcast; ++d ) \
			PACKM_SCAL_RO( ctypep_r, cha, chp, chp_r, mnk, mnk, scal2ris ); \
	} \
\
	/* invert the diagonal if requested */ \
	if ( invdiag ) \
	{ \
		/* TODO: real-only packing doesn't work for TRSM */ \
	} \
\
	/* if this an edge case in both directions, extend the diagonal with ones */ \
	for ( dim_t mnk = cdim; mnk < bli_min( cdim_max, n_max ); ++mnk ) \
	for ( dim_t d = 0; d < cdim_bcast; ++d ) \
		PACKM_SET_RO( chp_r, one, mnk ); \
}

GENTFUNC2R( scomplex, float,  c, s, scomplex, float,  c, s, packm_diag_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( scomplex, float,  c, s, dcomplex, double, z, d, packm_diag_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( dcomplex, double, z, d, scomplex, float,  c, s, packm_diag_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC2R( dcomplex, double, z, d, dcomplex, double, z, d, packm_diag_ro, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
