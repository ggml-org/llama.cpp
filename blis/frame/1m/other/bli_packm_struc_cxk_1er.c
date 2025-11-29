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
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       struc_t         strucc, \
       diag_t          diagc, \
       uplo_t          uploc, \
       conj_t          conjc, \
       pack_t          schema, \
       bool            invdiag, \
       dim_t           panel_dim, \
       dim_t           panel_len, \
       dim_t           panel_dim_max, \
       dim_t           panel_len_max, \
       dim_t           panel_dim_off, \
       dim_t           panel_len_off, \
       ctype* restrict kappa, \
       ctype* restrict c, inc_t incc, inc_t ldc, \
       ctype* restrict p,             inc_t ldp, \
                          inc_t is_p, \
       cntx_t*         cntx, \
       void*           params \
     ) \
{ \
	/* Handle micro-panel packing based on the structure of the matrix
	   being packed. */ \
	if      ( bli_is_general( strucc ) ) \
	{ \
		/* For micro-panels of general matrices, we can call the pack
		   kernel front-end directly. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc, \
		  schema, \
		  panel_dim, \
		  panel_dim_max, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  c, incc, ldc, \
		  p,       ldp, \
		  cntx \
		); \
	} \
	else if ( bli_is_herm_or_symm( strucc ) ) \
	{ \
		/* Call a helper function for micro-panels of Hermitian/symmetric
		   matrices. */ \
		PASTEMAC(ch,packm_herm_cxk_1er) \
		( \
		  strucc, \
		  diagc, \
		  uploc, \
		  conjc, \
		  schema, \
		  invdiag, \
		  panel_dim, \
		  panel_len, \
		  panel_dim_max, \
		  panel_len_max, \
		  panel_dim_off, \
		  panel_len_off, \
		  kappa, \
		  c, incc, ldc, \
		  p,       ldp, \
		     is_p, \
		  cntx, \
		  params \
		); \
	} \
	else /* ( bli_is_triangular( strucc ) ) */ \
	{ \
		/* Call a helper function for micro-panels of triangular
		   matrices. */ \
		PASTEMAC(ch,packm_tri_cxk_1er) \
		( \
		  strucc, \
		  diagc, \
		  uploc, \
		  conjc, \
		  schema, \
		  invdiag, \
		  panel_dim, \
		  panel_len, \
		  panel_dim_max, \
		  panel_len_max, \
		  panel_dim_off, \
		  panel_len_off, \
		  kappa, \
		  c, incc, ldc, \
		  p,       ldp, \
		     is_p, \
		  cntx, \
		  params \
		); \
	} \
}

INSERT_GENTFUNCCO_BASIC( packm_struc_cxk_1er, packm_cxk_1er )




#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       struc_t         strucc, \
       diag_t          diagc, \
       uplo_t          uploc, \
       conj_t          conjc, \
       pack_t          schema, \
       bool            invdiag, \
       dim_t           panel_dim, \
       dim_t           panel_len, \
       dim_t           panel_dim_max, \
       dim_t           panel_len_max, \
       dim_t           panel_dim_off, \
       dim_t           panel_len_off, \
       ctype* restrict kappa, \
       ctype* restrict c, inc_t incc, inc_t ldc, \
       ctype* restrict p,             inc_t ldp, \
                          inc_t is_p, \
       cntx_t*         cntx, \
       void*           params \
     ) \
{ \
	doff_t diagoffc = panel_dim_off - panel_len_off; \
	doff_t diagoffc_abs; \
	dim_t  j; \
\
	/* Handle the case where the micro-panel does NOT intersect the
	   diagonal separately from the case where it does intersect. */ \
	if ( !bli_intersects_diag_n( diagoffc, panel_dim, panel_len ) ) \
	{ \
		/* If the current panel is unstored, we need to make a few
		   adjustments so we refer to the data where it is actually
		   stored, also taking conjugation into account. (Note this
		   implicitly assumes we are operating on a dense panel
		   within a larger symmetric or Hermitian matrix, since a
		   general matrix would not contain any unstored region.) */ \
		if ( bli_is_unstored_subpart_n( diagoffc, uploc, panel_dim, panel_len ) ) \
		{ \
			c = c + diagoffc * ( doff_t )ldc + \
			       -diagoffc * ( doff_t )incc;  \
			bli_swap_incs( &incc, &ldc ); \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc ); \
		} \
\
		/* Pack the full panel. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc, \
		  schema, \
		  panel_dim, \
		  panel_dim_max, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  c, incc, ldc, \
		  p,       ldp, \
		  cntx  \
		); \
	} \
	else /* if ( bli_intersects_diag_n( diagoffc, panel_dim, panel_len ) ) */ \
	{ \
		ctype* restrict c10; \
		ctype* restrict p10; \
		dim_t           p10_dim, p10_len; \
		inc_t           incc10, ldc10; \
		doff_t          diagoffc10; \
		conj_t          conjc10; \
\
		ctype* restrict c12; \
		ctype* restrict p12; \
		dim_t           p12_dim, p12_len; \
		inc_t           incc12, ldc12; \
		doff_t          diagoffc12; \
		conj_t          conjc12; \
\
\
		/* Sanity check. Diagonals should not intersect the short end of
		   a micro-panel. If they do, then somehow the constraints on
		   cache blocksizes being a whole multiple of the register
		   blocksizes was somehow violated. */ \
		if ( diagoffc < 0 ) \
			bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
		diagoffc_abs = bli_abs( diagoffc ); \
\
		if      ( bli_is_lower( uploc ) ) \
		{ \
			p10_dim    = panel_dim; \
			p10_len    = diagoffc_abs; \
			p10        = p; \
			c10        = c; \
			incc10     = incc; \
			ldc10      = ldc; \
			conjc10    = conjc; \
\
			p12_dim    = panel_dim; \
			p12_len    = panel_len - p10_len; \
			j          = p10_len; \
			diagoffc12 = diagoffc_abs - j; \
			p12        = p + (j  )*ldp; \
			c12        = c + (j  )*ldc; \
			c12        = c12 + diagoffc12 * ( doff_t )ldc + \
			                  -diagoffc12 * ( doff_t )incc;  \
			incc12     = ldc; \
			ldc12      = incc; \
			conjc12    = conjc; \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc12 ); \
		} \
		else /* if ( bli_is_upper( uploc ) ) */ \
		{ \
			p10_dim    = panel_dim; \
			p10_len    = diagoffc_abs + panel_dim; \
			diagoffc10 = diagoffc; \
			p10        = p; \
			c10        = c; \
			c10        = c10 + diagoffc10 * ( doff_t )ldc + \
			                  -diagoffc10 * ( doff_t )incc;  \
			incc10     = ldc; \
			ldc10      = incc; \
			conjc10    = conjc; \
\
			p12_dim    = panel_dim; \
			p12_len    = panel_len - p10_len; \
			j          = p10_len; \
			p12        = p + (j  )*ldp; \
			c12        = c + (j  )*ldc; \
			incc12     = incc; \
			ldc12      = ldc; \
			conjc12    = conjc; \
\
			if ( bli_is_hermitian( strucc ) ) \
				bli_toggle_conj( &conjc10 ); \
		} \
\
		/* Pack to p10. For upper storage, this includes the unstored
		   triangle of c11. */ \
		/* NOTE: Since we're only packing partial panels here, we pass in
		   p1x_len as panel_len_max; otherwise, the packm kernel will zero-
		   fill the columns up to panel_len_max, which is not what we need
		   or want to happen. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc10, \
		  schema, \
		  p10_dim, \
		  panel_dim_max, \
		  p10_len, \
		  p10_len, \
		  kappa, \
		  c10, incc10, ldc10, \
		  p10,         ldp, \
		  cntx  \
		); \
\
		/* Pack to p12. For lower storage, this includes the unstored
		   triangle of c11. */ \
		/* NOTE: Since we're only packing partial panels here, we pass in
		   p1x_len as panel_len_max; otherwise, the packm kernel will zero-
		   fill the columns up to panel_len_max, which is not what we need
		   or want to happen. */ \
		PASTEMAC(ch,kername) \
		( \
		  conjc12, \
		  schema, \
		  p12_dim, \
		  panel_dim_max, \
		  p12_len, \
		  p12_len, \
		  kappa, \
		  c12, incc12, ldc12, \
		  p12,         ldp, \
		  cntx  \
		); \
\
		/* Pack the stored triangle of c11 to p11. */ \
		{ \
			dim_t           j   = diagoffc_abs; \
			ctype* restrict c11 = c + (j  )*ldc; \
			ctype* restrict p11 = p + (j  )*ldp; \
\
			PASTEMAC(ch,scal21ms_mxn_uplo) \
			( \
			  schema, \
			  uploc, \
			  conjc, \
			  panel_dim, \
			  kappa, \
			  c11, incc, ldc, \
			  p11,    1, ldp, ldp  \
			); \
\
			/* If we are packing a micro-panel with Hermitian structure,
			   we must take special care of the diagonal. Now, if kappa
			   were guaranteed to be unit, all we would need to do is
			   explicitly zero out the imaginary part of the diagonal of
			   p11, in case the diagonal of the source matrix contained
			   garbage (non-zero) imaginary values. HOWEVER, since kappa
			   can be non-unit, things become a little more complicated.
			   In general, we must re-apply the kappa scalar to ONLY the
			   real part of the diagonal of the source matrix and save
			   the result to the diagonal of p11. */ \
			if ( bli_is_hermitian( strucc ) ) \
			{ \
				ctype_r* restrict c11_r = ( ctype_r* )c11; \
				const dim_t       incc2 = 2*incc; \
				const dim_t       ldc2 = 2*ldc; \
\
				PASTEMAC3(ch,chr,ch,scal21ms_mxn_diag) \
				( \
				  schema, \
				  panel_dim, \
				  panel_dim, \
				  kappa, \
				  c11_r, incc2, ldc2, \
				  p11,   1,     ldp, ldp  \
				); \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC( packm_herm_cxk_1er, packm_cxk_1er )




#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       struc_t         strucc, \
       diag_t          diagc, \
       uplo_t          uploc, \
       conj_t          conjc, \
       pack_t          schema, \
       bool            invdiag, \
       dim_t           panel_dim, \
       dim_t           panel_len, \
       dim_t           panel_dim_max, \
       dim_t           panel_len_max, \
       dim_t           panel_dim_off, \
       dim_t           panel_len_off, \
       ctype* restrict kappa, \
       ctype* restrict c, inc_t incc, inc_t ldc, \
       ctype* restrict p,             inc_t ldp, \
                          inc_t is_p, \
       cntx_t*         cntx, \
       void*           params \
     ) \
{ \
	doff_t diagoffc     = panel_dim_off - panel_len_off; \
	doff_t diagoffc_abs = bli_abs( diagoffc ); \
	ctype* p11          = p + (diagoffc_abs  )*ldp; \
\
\
	/* Pack the panel. */ \
	PASTEMAC(ch,kername) \
	( \
	  conjc, \
	  schema, \
	  panel_dim, \
	  panel_dim_max, \
	  panel_len, \
	  panel_len_max, \
	  kappa, \
	  c, incc, ldc, \
	  p,       ldp, \
	  cntx  \
	); \
\
\
	/* Tweak the panel according to its triangular structure */ \
	{ \
		/* If the diagonal of c is implicitly unit, explicitly set the
		   the diagonal of the packed panel to kappa. */ \
		if ( bli_is_unit_diag( diagc ) ) \
		{ \
			PASTEMAC(ch,set1ms_mxn_diag) \
			( \
			  schema, \
			  0, \
			  0, \
			  panel_dim, \
			  panel_dim, \
			  kappa, \
			  p11, 1, ldp, ldp  \
			); \
		} \
\
\
		/* If requested, invert the diagonal of the packed panel. */ \
		if ( invdiag == TRUE ) \
		{ \
			PASTEMAC(ch,invert1ms_mxn_diag) \
			( \
			  schema, \
			  0, \
			  0, \
			  panel_dim, \
			  panel_dim, \
			  p11, 1, ldp, ldp  \
			); \
		} \
\
\
		/* Set the region opposite the diagonal of p to zero. To do this,
		   we need to reference the "unstored" region on the other side of
		   the diagonal. This amounts to toggling uploc and then shifting
		   the diagonal offset to shrink the newly referenced region (by
		   one diagonal). Note that this zero-filling is not needed for
		   trsm, since the unstored region is not referenced by the trsm
		   micro-kernel; however, zero-filling is needed for trmm, which
		   uses the gemm micro-kernel.*/ \
		{ \
			ctype* restrict zero         = PASTEMAC(ch,0); \
			uplo_t          uplop        = uploc; \
			doff_t          diagoffc11_0 = 0; \
			dim_t           p11_0_dim    = panel_dim - 1; \
\
			bli_toggle_uplo( &uplop ); \
			bli_shift_diag_offset_to_shrink_uplo( uplop, &diagoffc11_0 ); \
\
			/* Note that this macro works a little differently than the setm
			   operation. Here, we pass in the dimensions of only p11, rather
			   than the whole micro-panel, and furthermore we pass in the
			   "shrunken" dimensions of p11, corresponding to the toggling
			   and shrinking of the diagonal above. The macro will do the
			   right thing, incrementing the pointer to p11 by the appropriate
			   leading dimension (ldp or rs_p), and setting only the lower
			   or upper triangle to zero. */ \
			PASTEMAC(ch,set1ms_mxn_uplo) \
			( \
			  schema, \
			  diagoffc11_0, \
			  uplop, \
			  p11_0_dim, \
			  p11_0_dim, \
			  zero, \
			  p11, 1, ldp, ldp  \
			); \
		} \
	} \
\
	/* If this micro-panel is an edge case in both panel dimension and
	   length, then it must be a bottom-right corner case, which
	   typically only happens for micro-panels being packed for trsm.
	   (It also happens for trmm if kr > 1.) Here, we set the part of
	   the diagonal that extends into the zero-padded region to
	   identity. This prevents NaNs and Infs from creeping into the
	   computation. If this code does execute for trmm, it is okay,
	   because those 1.0's that extend into the bottom-right region
	   end up getting muliplied by the 0.0's in the zero-padded region
	   of the other matrix. */ \
	if ( panel_dim != panel_dim_max && \
	     panel_len != panel_len_max ) \
	{ \
		ctype* restrict one    = PASTEMAC(ch,1); \
		dim_t           offm   = panel_dim; \
		dim_t           offn   = panel_len; \
		dim_t           m_edge = panel_dim_max - panel_dim; \
		dim_t           n_edge = panel_len_max - panel_len; \
\
		PASTEMAC(ch,set1ms_mxn_diag) \
		( \
		  schema, \
		  offm, \
		  offn, \
		  m_edge, \
		  n_edge, \
		  one, \
		  p, 1, ldp, ldp  \
		); \
	} \
}

INSERT_GENTFUNCCO_BASIC( packm_tri_cxk_1er, packm_cxk_1er )

