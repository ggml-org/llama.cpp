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

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname) \
     ( \
       doff_t  diagoffx, \
       diag_t  diagx, \
       uplo_t  uplox, \
       trans_t transx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       ctype*  y, inc_t rs_y, inc_t cs_y, \
       cntx_t* cntx \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	uplo_t   uplox_eff; \
	conj_t   conjx; \
	dim_t    n_iter; \
	dim_t    n_elem_max; \
	inc_t    ldx, incx; \
	inc_t    ldy, incy; \
	dim_t    ij0, n_shift; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_2m \
	( \
	  diagoffx, diagx, transx, \
	  uplox, m, n, rs_x, cs_x, rs_y, cs_y, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, &incy, &ldy, \
	  &ij0, &n_shift \
	); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	/* Extract the conjugation component from the transx parameter. */ \
	conjx = bli_extract_conj( transx ); \
\
	/* Query the kernel needed for this operation. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( dim_t j = 0; j < n_iter; ++j ) \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			ctype* x1 = x + (j  )*ldx + (0  )*incx; \
			ctype* y1 = y + (j  )*ldy + (0  )*incy; \
\
			/* Invoke the kernel with the appropriate parameters. */ \
			f \
			( \
			  conjx, \
			  n_elem, \
			  x1, incx, \
			  y1, incy, \
			  cntx  \
			); \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				ctype* x1 = x + (ij0+j  )*ldx + (0  )*incx; \
				ctype* y1 = y + (ij0+j  )*ldy + (0  )*incy; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjx, \
				  n_elem, \
				  x1, incx, \
				  y1, incy, \
				  cntx  \
				); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t offi   = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				const dim_t n_elem = n_elem_max - offi; \
\
				ctype* x1 = x + (j  )*ldx + (ij0+offi  )*incx; \
				ctype* y1 = y + (j  )*ldy + (ij0+offi  )*incy; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjx, \
				  n_elem, \
				  x1, incx, \
				  y1, incy, \
				  cntx  \
				); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( addm_unb_var1,  addv,  BLIS_ADDV_KER )
INSERT_GENTFUNC_BASIC( copym_unb_var1, copyv, BLIS_COPYV_KER )
INSERT_GENTFUNC_BASIC( subm_unb_var1,  subv,  BLIS_SUBV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname) \
     ( \
       doff_t  diagoffx, \
       diag_t  diagx, \
       uplo_t  uplox, \
       trans_t transx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       ctype*  y, inc_t rs_y, inc_t cs_y, \
       cntx_t* cntx \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	uplo_t   uplox_eff; \
	conj_t   conjx; \
	dim_t    n_iter; \
	dim_t    n_elem_max; \
	inc_t    ldx, incx; \
	inc_t    ldy, incy; \
	dim_t    ij0, n_shift; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_2m \
	( \
	  diagoffx, diagx, transx, \
	  uplox, m, n, rs_x, cs_x, rs_y, cs_y, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, &incy, &ldy, \
	  &ij0, &n_shift \
	); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	/* Extract the conjugation component from the transx parameter. */ \
	conjx = bli_extract_conj( transx ); \
\
	/* Query the kernel needed for this operation. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( dim_t j = 0; j < n_iter; ++j ) \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			ctype* x1 = x + (j  )*ldx + (0  )*incx; \
			ctype* y1 = y + (j  )*ldy + (0  )*incy; \
\
			/* Invoke the kernel with the appropriate parameters. */ \
			f \
			( \
			  conjx, \
			  n_elem, \
			  alpha, \
			  x1, incx, \
			  y1, incy, \
			  cntx  \
			); \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				ctype* x1 = x + (ij0+j  )*ldx + (0  )*incx; \
				ctype* y1 = y + (ij0+j  )*ldy + (0  )*incy; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjx, \
				  n_elem, \
				  alpha, \
				  x1, incx, \
				  y1, incy, \
				  cntx  \
				); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t offi   = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				const dim_t n_elem = n_elem_max - offi; \
\
				ctype* x1 = x + (j  )*ldx + (ij0+offi  )*incx; \
				ctype* y1 = y + (j  )*ldy + (ij0+offi  )*incy; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjx, \
				  n_elem, \
				  alpha, \
				  x1, incx, \
				  y1, incy, \
				  cntx  \
				); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( axpym_unb_var1,  axpyv,  BLIS_AXPYV_KER )
INSERT_GENTFUNC_BASIC( scal2m_unb_var1, scal2v, BLIS_SCAL2V_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname) \
     ( \
       conj_t  conjalpha, \
       doff_t  diagoffx, \
       diag_t  diagx, \
       uplo_t  uplox, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       cntx_t* cntx \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	uplo_t   uplox_eff; \
	dim_t    n_iter; \
	dim_t    n_elem_max; \
	inc_t    ldx, incx; \
	dim_t    ij0, n_shift; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_1m \
	( \
	  diagoffx, diagx, \
	  uplox, m, n, rs_x, cs_x, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, \
	  &ij0, &n_shift \
	); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	/* Query the kernel needed for this operation. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( dim_t j = 0; j < n_iter; ++j ) \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			ctype* x1 = x + (j  )*ldx + (0  )*incx; \
\
			/* Invoke the kernel with the appropriate parameters. */ \
			f \
			( \
			  conjalpha, \
			  n_elem, \
			  alpha, \
			  x1, incx, \
			  cntx  \
			); \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				ctype* x1 = x + (ij0+j  )*ldx + (0  )*incx; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjalpha, \
				  n_elem, \
				  alpha, \
				  x1, incx, \
				  cntx  \
				); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t offi   = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				const dim_t n_elem = n_elem_max - offi; \
\
				ctype* x1 = x + (j  )*ldx + (ij0+offi  )*incx; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjalpha, \
				  n_elem, \
				  alpha, \
				  x1, incx, \
				  cntx  \
				); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( invscalm_unb_var1, invscalv, BLIS_INVSCALV_KER )
INSERT_GENTFUNC_BASIC( scalm_unb_var1, scalv, BLIS_SCALV_KER )
INSERT_GENTFUNC_BASIC( setm_unb_var1,  setv,  BLIS_SETV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname) \
     ( \
       doff_t  diagoffx, \
       diag_t  diagx, \
       uplo_t  uplox, \
       trans_t transx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       ctype*  beta, \
       ctype*  y, inc_t rs_y, inc_t cs_y, \
       cntx_t* cntx \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	uplo_t   uplox_eff; \
	conj_t   conjx; \
	dim_t    n_iter; \
	dim_t    n_elem_max; \
	inc_t    ldx, incx; \
	inc_t    ldy, incy; \
	dim_t    ij0, n_shift; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_2m \
	( \
	  diagoffx, diagx, transx, \
	  uplox, m, n, rs_x, cs_x, rs_y, cs_y, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, &incy, &ldy, \
	  &ij0, &n_shift \
	); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	/* Extract the conjugation component from the transx parameter. */ \
	conjx = bli_extract_conj( transx ); \
\
	/* Query the kernel needed for this operation. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( dim_t j = 0; j < n_iter; ++j ) \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			ctype* x1 = x + (j  )*ldx + (0  )*incx; \
			ctype* y1 = y + (j  )*ldy + (0  )*incy; \
\
			/* Invoke the kernel with the appropriate parameters. */ \
			f \
			( \
			  conjx, \
			  n_elem, \
			  x1, incx, \
			  beta, \
			  y1, incy, \
			  cntx  \
			); \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				ctype* x1 = x + (ij0+j  )*ldx + (0  )*incx; \
				ctype* y1 = y + (ij0+j  )*ldy + (0  )*incy; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjx, \
				  n_elem, \
				  x1, incx, \
				  beta, \
				  y1, incy, \
				  cntx  \
				); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t offi   = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				const dim_t n_elem = n_elem_max - offi; \
\
				ctype* x1 = x + (j  )*ldx + (ij0+offi  )*incx; \
				ctype* y1 = y + (j  )*ldy + (ij0+offi  )*incy; \
\
				/* Invoke the kernel with the appropriate parameters. */ \
				f \
				( \
				  conjx, \
				  n_elem, \
				  x1, incx, \
				  beta, \
				  y1, incy, \
				  cntx  \
				); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( xpbym_unb_var1,  xpbyv,  BLIS_XPBYV_KER )


#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_y, chx, chy, opname ) \
\
void PASTEMAC(chx,chy,opname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       trans_t  transx, \
       dim_t    m, \
       dim_t    n, \
       ctype_x* x, inc_t rs_x, inc_t cs_x, \
       ctype_y* beta, \
       ctype_y* y, inc_t rs_y, inc_t cs_y, \
       cntx_t*  cntx \
     ) \
{ \
	uplo_t uplox_eff; \
	dim_t  n_iter; \
	dim_t  n_elem_max; \
	inc_t  ldx, incx; \
	inc_t  ldy, incy; \
	dim_t  ij0, n_shift; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_2m \
	( \
	  diagoffx, diagx, transx, \
	  uplox, m, n, rs_x, cs_x, rs_y, cs_y, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, &incy, &ldy, \
	  &ij0, &n_shift \
	); \
\
	/* Extract the conjugation component from the transx parameter. */ \
	/*conjx = bli_extract_conj( transx );*/ \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_teq1s( chy, *beta ) ) \
	{ \
		if ( incx == 1 && incy == 1 ) \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				ctype_x* restrict x1 = x + (j  )*ldx + (0  )*incx; \
				ctype_y* restrict y1 = y + (j  )*ldy + (0  )*incy; \
\
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					bli_tadds( chx,chy,chy, x1[i], y1[i] ); \
				} \
			} \
		} \
		else \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				ctype_x* restrict x1 = x + (j  )*ldx + (0  )*incx; \
				ctype_y* restrict y1 = y + (j  )*ldy + (0  )*incy; \
\
				ctype_x* restrict chi1 = x1; \
				ctype_y* restrict psi1 = y1; \
\
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					bli_tadds( chx,chy,chy, *chi1, *psi1 ); \
\
					chi1 += incx; \
					psi1 += incy; \
				} \
			} \
		} \
	} \
	else /* ( !bli_teq1s( chy, *beta ) ) */ \
	{ \
		if ( incx == 1 && incy == 1 ) \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				ctype_x* restrict x1 = x + (j  )*ldx + (0  )*incx; \
				ctype_y* restrict y1 = y + (j  )*ldy + (0  )*incy; \
\
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					bli_txpbys( chx,chy,chy,chy, x1[i], *beta, y1[i] ); \
				} \
			} \
		} \
		else \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				ctype_x* restrict x1 = x + (j  )*ldx + (0  )*incx; \
				ctype_y* restrict y1 = y + (j  )*ldy + (0  )*incy; \
\
				ctype_x* restrict chi1 = x1; \
				ctype_y* restrict psi1 = y1; \
\
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					bli_txpbys( chx,chy,chy,chy, *chi1, *beta, *psi1 ); \
\
					chi1 += incx; \
					psi1 += incy; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNC2_BASIC( xpbym_md_unb_var1 )
INSERT_GENTFUNC2_MIX_DP( xpbym_md_unb_var1 )

