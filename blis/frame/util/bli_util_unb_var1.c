/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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
#include <fenv.h>

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* asum, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype*  chi1; \
	ctype_r chi1_r; \
	ctype_r chi1_i; \
	ctype_r absum; \
	dim_t   i; \
\
	/* Initialize the absolute sum accumulator to zero. */ \
	bli_tset0s( chr, absum ); \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		chi1 = x + (i  )*incx; \
\
		/* Get the real and imaginary components of chi1. */ \
		bli_tgets( ch,chr, *chi1, chi1_r, chi1_i ); \
\
		/* Replace chi1_r and chi1_i with their absolute values. */ \
		chi1_r = bli_fabs( chi1_r ); \
		chi1_i = bli_fabs( chi1_i ); \
\
		/* Accumulate the real and imaginary components into absum. */ \
		bli_tadds( chr,chr,chr, chi1_r, absum ); \
		bli_tadds( chr,chr,chr, chi1_i, absum ); \
	} \
\
	/* Store the final value of absum to the output variable. */ \
	bli_tcopys( chr,chr, absum, *asum ); \
}

INSERT_GENTFUNCR_BASIC( asumv_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	ctype_r* zeror = PASTEMAC(chr,0); \
	doff_t   diagoffa; \
\
	/* If the dimension is zero, return early. */ \
	if ( bli_zero_dim1( m ) ) return; \
\
	/* In order to avoid the main diagonal, we must nudge the diagonal either
	   up or down by one, depending on which triangle is currently stored. */ \
	if        ( bli_is_upper( uploa ) )   diagoffa =  1; \
	else /*if ( bli_is_lower( uploa ) )*/ diagoffa = -1; \
\
	/* We will be reflecting the stored region over the diagonal into the
	   unstored region, so a transposition is necessary. Furthermore, since
	   we are creating a Hermitian matrix, we must also conjugate. */ \
	PASTEMAC(ch,copym,BLIS_TAPI_EX_SUF) \
	( \
	  diagoffa, \
	  BLIS_NONUNIT_DIAG, \
	  uploa, \
	  BLIS_CONJ_TRANSPOSE, \
	  m, \
	  m, \
	  a, rs_a, cs_a, \
	  a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
\
	/* Set the imaginary parts of the diagonal elements to zero. */ \
	PASTEMAC(ch,setid,BLIS_TAPI_EX_SUF) \
	( \
	  0, \
	  m, \
	  m, \
	  zeror, \
	  a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNCR_BASIC( mkherm_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	doff_t  diagoffa; \
\
	/* If the dimension is zero, return early. */ \
	if ( bli_zero_dim1( m ) ) return; \
\
	/* In order to avoid the main diagonal, we must nudge the diagonal either
	   up or down by one, depending on which triangle is currently stored. */ \
	if        ( bli_is_upper( uploa ) )   diagoffa =  1; \
	else /*if ( bli_is_lower( uploa ) )*/ diagoffa = -1; \
\
	/* We will be reflecting the stored region over the diagonal into the
	   unstored region, so a transposition is necessary. */ \
	PASTEMAC(ch,copym,BLIS_TAPI_EX_SUF) \
	( \
	  diagoffa, \
	  BLIS_NONUNIT_DIAG, \
	  uploa, \
	  BLIS_TRANSPOSE, \
	  m, \
	  m, \
	  a, rs_a, cs_a, \
	  a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( mksymm_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	ctype*  zero = PASTEMAC(ch,0); \
	doff_t  diagoffa; \
\
	/* If the dimension is zero, return early. */ \
	if ( bli_zero_dim1( m ) ) return; \
\
	/* Toggle uplo so that it refers to the unstored triangle. */ \
	bli_toggle_uplo( &uploa ); \
\
	/* In order to avoid the main diagonal, we must nudge the diagonal either
	   up or down by one, depending on which triangle is to be zeroed. */ \
	if        ( bli_is_upper( uploa ) )   diagoffa =  1; \
	else /*if ( bli_is_lower( uploa ) )*/ diagoffa = -1; \
\
	/* Set the unstored triangle to zero. */ \
	PASTEMAC(ch,setm,BLIS_TAPI_EX_SUF) \
	( \
	  BLIS_NO_CONJUGATE, \
	  diagoffa, \
	  BLIS_NONUNIT_DIAG, \
	  uploa, \
	  m, \
	  m, \
	  zero, \
	  a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( mktrim_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype*  chi1; \
	ctype_r abs_chi1; \
	ctype_r absum; \
	dim_t   i; \
\
	/* Initialize the absolute sum accumulator to zero. */ \
	bli_tset0s( chr, absum ); \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		chi1 = x + (i  )*incx; \
\
		/* Compute the absolute value (or complex magnitude) of chi1. */ \
		bli_tabval2s( ch,chr,chr, *chi1, abs_chi1 ); \
\
		/* Accumulate the absolute value of chi1 into absum. */ \
		bli_tadds( chr,chr,chr, abs_chi1, absum ); \
	} \
\
	/* Store final value of absum to the output variable. */ \
	bli_tcopys( chr,chr, absum, *norm ); \
}

INSERT_GENTFUNCR_BASIC( norm1v_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype_r* zero       = PASTEMAC(chr,0); \
	ctype_r* one        = PASTEMAC(chr,1); \
	ctype_r  scale; \
	ctype_r  sumsq; \
	ctype_r  sqrt_sumsq; \
\
	/* Initialize scale and sumsq to begin the summation. */ \
	bli_tcopys( chr,chr, *zero, scale ); \
	bli_tcopys( chr,chr, *one,  sumsq ); \
\
	/* Compute the sum of the squares of the vector. */ \
	PASTEMAC(ch,kername) \
	( \
	  n, \
	  x, incx, \
	  &scale, \
	  &sumsq, \
	  cntx, \
	  rntm  \
	); \
\
	/* Compute: norm = scale * sqrt( sumsq ) */ \
	bli_tsqrt2s( chr,chr,chr, sumsq, sqrt_sumsq ); \
	bli_tscals( chr,chr,chr, scale, sqrt_sumsq ); \
\
	/* Store the final value to the output variable. */ \
	bli_tcopys( chr,chr, sqrt_sumsq, *norm ); \
}

//INSERT_GENTFUNCR_BASIC( normfv_unb_var1, sumsqv_unb_var1 )
GENTFUNCR( scomplex, float,  c, s, normfv_unb_var1, sumsqv_unb_var1 )
GENTFUNCR( dcomplex, double, z, d, normfv_unb_var1, sumsqv_unb_var1 )

#undef  GENTFUNCR
// We've disabled the dotv-based implementation because that method of
// computing the sum of the squares of x inherently does not check for
// overflow. Instead, we use the fallback method based on sumsqv, which
// takes care to not overflow unnecessarily (ie: takes care for the
// sqrt( sum of the squares of x ) to not overflow if the sum of the
// squares of x would normally overflow. See GitHub issue #332 for
// discussion.
#if 0 //defined(FE_OVERFLOW) && !defined(__APPLE__)
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype_r* zero       = PASTEMAC(chr,0); \
	ctype_r* one        = PASTEMAC(chr,1); \
	ctype_r  scale; \
	ctype_r  sumsq; \
	ctype_r  sqrt_sumsq; \
\
	/* Initialize scale and sumsq to begin the summation. */ \
	bli_tcopys( chr, *zero, scale ); \
	bli_tcopys( chr, *one,  sumsq ); \
\
	/* An optimization: first try to use dotv to compute the sum of
	   the squares of the vector. If no floating-point exceptions
	   (specifically, overflow and invalid exceptions) were produced,
	   then we accept the computed value and returne early. The cost
	   of this optimization is the "sunk" cost of the initial dotv
	   when sumsqv must be used instead. However, we expect that the
	   vast majority of use cases will not produce exceptions, and
	   therefore only one pass through the data, via dotv, will be
	   required. */ \
	if ( TRUE ) \
	{ \
		int      f_exp_raised;\
		ctype    sumsqc; \
\
		feclearexcept( FE_ALL_EXCEPT );\
\
		PASTEMAC(ch,dotv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  BLIS_NO_CONJUGATE, \
		  n,\
		  x, incx, \
		  x, incx, \
		  &sumsqc, \
		  cntx, \
		  rntm  \
		); \
\
		bli_tcopys( ch,chr, sumsqc, sumsq ); \
\
		f_exp_raised = fetestexcept( FE_OVERFLOW | FE_INVALID );\
\
		if ( !f_exp_raised ) \
		{ \
		    tsqrt2s( chr, sumsq, *norm ); \
		    return; \
		} \
	} \
\
	/* Compute the sum of the squares of the vector. */ \
	PASTEMAC(ch,kername) \
	( \
	  n, \
	  x, incx, \
	  &scale, \
	  &sumsq, \
	  cntx, \
	  rntm  \
	); \
\
	/* Compute: norm = scale * sqrt( sumsq ) */ \
	tsqrt2s( chr, sumsq, sqrt_sumsq ); \
	bli_tscals( chr, scale, sqrt_sumsq ); \
\
	/* Store the final value to the output variable. */ \
	bli_tcopys( chr, sqrt_sumsq, *norm ); \
}
#else
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype_r* zero       = PASTEMAC(chr,0); \
	ctype_r* one        = PASTEMAC(chr,1); \
	ctype_r  scale; \
	ctype_r  sumsq; \
	ctype_r  sqrt_sumsq; \
\
	/* Initialize scale and sumsq to begin the summation. */ \
	bli_tcopys( chr,chr, *zero, scale ); \
	bli_tcopys( chr,chr, *one,  sumsq ); \
\
	/* Compute the sum of the squares of the vector. */ \
\
	PASTEMAC(ch,kername) \
	( \
	  n, \
	  x, incx, \
	  &scale, \
	  &sumsq, \
	  cntx, \
	  rntm  \
	); \
\
	/* Compute: norm = scale * sqrt( sumsq ) */ \
	bli_tsqrt2s( chr,chr,chr, sumsq, sqrt_sumsq ); \
	bli_tscals( chr,chr,chr, scale, sqrt_sumsq ); \
\
	/* Store the final value to the output variable. */ \
	bli_tcopys( chr,chr, sqrt_sumsq, *norm ); \
}
#endif
GENTFUNCR( float,   float,  s, s, normfv_unb_var1, sumsqv_unb_var1 )
GENTFUNCR( double,  double, d, d, normfv_unb_var1, sumsqv_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype*  chi1; \
	ctype_r abs_chi1; \
	ctype_r abs_chi1_max; \
	dim_t   i; \
\
	/* Initialize the maximum absolute value to zero. */ \
	bli_tset0s( chr, abs_chi1_max ); \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		chi1 = x + (i  )*incx; \
\
		/* Compute the absolute value (or complex magnitude) of chi1. */ \
		bli_tabval2s( ch,chr,chr, *chi1, abs_chi1 ); \
\
		/* If the absolute value of the current element exceeds that of
		   the previous largest, save it and its index. If NaN is
		   encountered, then treat it the same as if it were a valid
		   value that was larger than any previously seen. This
		   behavior mimics that of LAPACK's ?lange(). */ \
		if ( abs_chi1_max < abs_chi1 || PASTEMAC(chr,isnan)( abs_chi1 ) ) \
		{ \
			bli_tcopys( chr,chr, abs_chi1, abs_chi1_max ); \
		} \
	} \
\
	/* Store the final value to the output variable. */ \
	bli_tcopys( chr,chr, abs_chi1_max, *norm ); \
}

INSERT_GENTFUNCR_BASIC( normiv_unb_var1 )



#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype*  one = PASTEMAC(ch,1); \
	ctype*  x0; \
	ctype*  chi1; \
	ctype*  x2; \
	ctype_r absum_max; \
	ctype_r absum_j; \
	ctype_r abval_chi1; \
	uplo_t  uplox_eff; \
	dim_t   n_iter; \
	dim_t   n_elem, n_elem_max; \
	inc_t   ldx, incx; \
	dim_t   j, i; \
	dim_t   ij0, n_shift; \
\
	/* Initialize the maximum absolute column sum to zero. */ \
	bli_tset0s( chr, absum_max ); \
\
	/* If either dimension is zero, return with absum_max equal to zero. */ \
	if ( bli_zero_dim2( m, n ) ) \
	{ \
		bli_tcopys( chr,chr, absum_max, *norm ); \
		return; \
	} \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_1m_noswap \
	( \
	  diagoffx, BLIS_NONUNIT_DIAG, \
	  uplox, m, n, rs_x, cs_x, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, \
	  &ij0, &n_shift \
	); \
\
	/* If the matrix is zeros, return with absum_max equal to zero. */ \
	if ( bli_is_zeros( uplox_eff ) ) \
	{ \
		bli_tcopys( chr,chr, absum_max, *norm ); \
		return; \
	} \
\
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( j = 0; j < n_iter; ++j ) \
		{ \
			n_elem = n_elem_max; \
\
			x0     = x + (j  )*ldx + (0  )*incx; \
\
			/* Compute the norm of the current column. */ \
			PASTEMAC(ch,kername) \
			( \
			  n_elem, \
			  x0, incx, \
			  &absum_j, \
			  cntx, \
			  rntm  \
			); \
\
			/* If absum_j is greater than the previous maximum value,
			   then save it. */ \
			if ( absum_max < absum_j || PASTEMAC(chr,isnan)( absum_j ) ) \
			{ \
				bli_tcopys( chr,chr, absum_j, absum_max ); \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				x0     = x + (ij0+j  )*ldx + (0       )*incx; \
				chi1   = x + (ij0+j  )*ldx + (n_elem-1)*incx; \
\
				/* Compute the norm of the super-diagonal elements. */ \
				PASTEMAC(ch,kername) \
				( \
				  n_elem - 1, \
				  x0, incx, \
				  &absum_j, \
				  cntx, \
				  rntm  \
				); \
\
				if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
				/* Handle the diagonal element separately in case it's
				   unit. */ \
				bli_tabval2s( ch,chr,chr, *chi1, abval_chi1 ); \
				bli_tadds( chr,chr,chr, abval_chi1, absum_j ); \
\
				/* If absum_j is greater than the previous maximum value,
				   then save it. */ \
				if ( absum_max < absum_j || PASTEMAC(chr,isnan)( absum_j ) ) \
				{ \
					bli_tcopys( chr,chr, absum_j, absum_max ); \
				} \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				n_elem = n_elem_max - i; \
\
				chi1   = x + (j  )*ldx + (ij0+i  )*incx; \
				x2     = x + (j  )*ldx + (ij0+i+1)*incx; \
\
				/* Compute the norm of the sub-diagonal elements. */ \
				PASTEMAC(ch,kername) \
				( \
				  n_elem - 1, \
				  x2, incx, \
				  &absum_j, \
				  cntx, \
				  rntm  \
				); \
\
				if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
				/* Handle the diagonal element separately in case it's
				   unit. */ \
				bli_tabval2s( ch,chr,chr, *chi1, abval_chi1 ); \
				bli_tadds( chr,chr,chr, abval_chi1, absum_j ); \
\
				/* If absum_j is greater than the previous maximum value,
				   then save it. */ \
				if ( absum_max < absum_j || PASTEMAC(chr,isnan)( absum_j ) ) \
				{ \
					bli_tcopys( chr,chr, absum_j, absum_max ); \
				} \
			} \
		} \
	} \
\
	/* Store final value of absum_max to the output variable. */ \
	bli_tcopys( chr,chr, absum_max, *norm ); \
}

INSERT_GENTFUNCR_BASIC( norm1m_unb_var1, norm1v_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype*   one    = PASTEMAC(ch,1); \
	ctype_r* one_r  = PASTEMAC(chr,1); \
	ctype_r* zero_r = PASTEMAC(chr,0); \
	ctype*   x0; \
	ctype*   chi1; \
	ctype*   x2; \
	ctype_r  scale; \
	ctype_r  sumsq; \
	ctype_r  sqrt_sumsq; \
	uplo_t   uplox_eff; \
	dim_t    n_iter; \
	dim_t    n_elem, n_elem_max; \
	inc_t    ldx, incx; \
	dim_t    j, i; \
	dim_t    ij0, n_shift; \
\
	/* Return a norm of zero if either dimension is zero. */ \
	if ( bli_zero_dim2( m, n ) ) \
	{ \
		bli_tset0s( chr, *norm ); \
		return; \
	} \
\
	/* Set various loop parameters. Here, we pretend that diagx is equal to
	   BLIS_NONUNIT_DIAG because we handle the unit diagonal case manually. */ \
	bli_set_dims_incs_uplo_1m \
	( \
	  diagoffx, BLIS_NONUNIT_DIAG, \
	  uplox, m, n, rs_x, cs_x, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, \
	  &ij0, &n_shift \
	); \
\
	/* Check the effective uplo; if it's zeros, then our norm is zero. */ \
	if ( bli_is_zeros( uplox_eff ) ) \
	{ \
		bli_tset0s( chr, *norm ); \
		return; \
	} \
\
	/* Initialize scale and sumsq to begin the summation. */ \
	bli_tcopys( chr,chr, *zero_r, scale ); \
	bli_tcopys( chr,chr, *one_r,  sumsq ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( j = 0; j < n_iter; ++j ) \
		{ \
			n_elem = n_elem_max; \
\
			x0     = x + (j  )*ldx + (0  )*incx; \
\
			/* Compute the norm of the current column. */ \
			PASTEMAC(ch,kername) \
			( \
			  n_elem, \
			  x0, incx, \
			  &scale, \
			  &sumsq, \
			  cntx, \
			  rntm  \
			); \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				x0     = x + (ij0+j  )*ldx + (0       )*incx; \
				chi1   = x + (ij0+j  )*ldx + (n_elem-1)*incx; \
\
				/* Sum the squares of the super-diagonal elements. */ \
				PASTEMAC(ch,kername) \
				( \
				  n_elem - 1, \
				  x0, incx, \
				  &scale, \
				  &sumsq, \
				  cntx, \
				  rntm  \
				); \
\
				if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
				/* Handle the diagonal element separately in case it's
				   unit. */ \
				PASTEMAC(ch,kername) \
				( \
				  1, \
				  chi1, incx, \
				  &scale, \
				  &sumsq, \
				  cntx, \
				  rntm  \
				); \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				n_elem = n_elem_max - i; \
\
				chi1   = x + (j  )*ldx + (ij0+i  )*incx; \
				x2     = x + (j  )*ldx + (ij0+i+1)*incx; \
\
				/* Sum the squares of the sub-diagonal elements. */ \
				PASTEMAC(ch,kername) \
				( \
				  n_elem - 1, \
				  x2, incx, \
				  &scale, \
				  &sumsq, \
				  cntx, \
				  rntm  \
				); \
\
				if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
				/* Handle the diagonal element separately in case it's
				   unit. */ \
				PASTEMAC(ch,kername) \
				( \
				  1, \
				  chi1, incx, \
				  &scale, \
				  &sumsq, \
				  cntx, \
				  rntm  \
				); \
			} \
		} \
	} \
\
	/* Compute: norm = scale * sqrt( sumsq ) */ \
	bli_tsqrt2s( chr,chr,chr, sumsq, sqrt_sumsq ); \
	bli_tscals( chr,chr,chr, scale, sqrt_sumsq ); \
\
	/* Store the final value to the output variable. */ \
	bli_tcopys( chr,chr, sqrt_sumsq, *norm ); \
}

INSERT_GENTFUNCR_BASIC( normfm_unb_var1, sumsqv_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	/* Induce a transposition so that rows become columns. */ \
	bli_swap_dims( &m, &n ); \
	bli_swap_incs( &rs_x, &cs_x ); \
	bli_toggle_uplo( &uplox ); \
	bli_negate_diag_offset( &diagoffx ); \
\
	/* Now we can simply compute the 1-norm of this transposed matrix,
	   which will be equivalent to the infinity-norm of the original
	   matrix. */ \
	PASTEMAC(ch,kername) \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  m, \
	  n, \
	  x, rs_x, cs_x, \
	  norm, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNCR_BASIC( normim_unb_var1, norm1m_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, randmac ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	ctype* chi1; \
	dim_t  i; \
\
	chi1 = x; \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		PASTEMAC(t,randmac)( ch, *chi1 ); \
\
		chi1 += incx; \
	} \
}

INSERT_GENTFUNC_BASIC( randv_unb_var1,  rands )
INSERT_GENTFUNC_BASIC( randnv_unb_var1, randnp2s )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t  diagoffx, \
       uplo_t  uplox, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	ctype* one = PASTEMAC(ch,1); \
	ctype* x0; \
	ctype* x1; \
	ctype* x2; \
	ctype* chi1; \
	ctype  beta; \
	ctype  omega; \
	double max_m_n; \
	uplo_t uplox_eff; \
	dim_t  n_iter; \
	dim_t  n_elem, n_elem_max; \
	inc_t  ldx, incx; \
	dim_t  j, i; \
	dim_t  ij0, n_shift; \
\
	/* Set various loop parameters. Here, we pretend that diagx is equal to
	   BLIS_NONUNIT_DIAG because we handle the unit diagonal case manually. */ \
	bli_set_dims_incs_uplo_1m \
	( \
	  diagoffx, BLIS_NONUNIT_DIAG, \
	  uplox, m, n, rs_x, cs_x, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, \
	  &ij0, &n_shift \
	); \
\
	if ( bli_is_zeros( uplox_eff ) ) return; \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( j = 0; j < n_iter; ++j ) \
		{ \
			n_elem = n_elem_max; \
\
			x1     = x + (j  )*ldx + (0  )*incx; \
\
			/*PASTEMAC(ch,kername,BLIS_TAPI_EX_SUF)*/ \
			PASTEMAC(ch,kername) \
			( \
			  n_elem, \
			  x1, incx, \
			  cntx, \
			  rntm  \
			); \
		} \
	} \
	else \
	{ \
		max_m_n = bli_max( m, n ); \
\
		bli_tsets( d,ch, max_m_n, 0.0, omega ); \
		bli_tcopys( ch,ch, *one, beta ); \
		bli_tinvscals( ch,ch,ch, omega, beta ); \
\
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				x1     = x + (ij0+j  )*ldx + (0  )*incx; \
				x0     = x1; \
				chi1   = x1 + (n_elem-1)*incx; \
\
				/*PASTEMAC(ch,kername,BLIS_TAPI_EX_SUF)*/ \
				PASTEMAC(ch,kername) \
				( \
				  n_elem, \
				  x1, incx, \
				  cntx, \
				  rntm  \
				); \
\
				( void )x0; \
				( void )chi1; \
				/* We want positive diagonal elements between 1 and 2. */ \
/*
				bli_tabval2s( ch,ch,ch, *chi1, *chi1 ); \
				bli_tadds( ch,ch,ch, *one, *chi1 ); \
*/ \
\
				/* Scale the super-diagonal elements by 1/max(m,n). */ \
/*
				PASTEMAC(ch,scalv) \
				( \
				  BLIS_NO_CONJUGATE, \
				  n_elem - 1, \
				  &beta, \
				  x0, incx, \
				  cntx  \
				); \
*/ \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				n_elem = n_elem_max - i; \
\
				x1     = x + (j  )*ldx + (ij0+i  )*incx; \
				x2     = x1 + incx; \
				chi1   = x1; \
\
				/*PASTEMAC(ch,kername,BLIS_TAPI_EX_SUF)*/ \
				PASTEMAC(ch,kername) \
				( \
				  n_elem, \
				  x1, incx, \
				  cntx, \
				  rntm  \
				); \
\
				( void )x2; \
				( void )chi1; \
				/* We want positive diagonal elements between 1 and 2. */ \
/*
				bli_tabval2s( ch,ch,ch, *chi1, *chi1 ); \
				bli_tadds( ch,ch,ch, *one, *chi1 ); \
*/ \
\
				/* Scale the sub-diagonal elements by 1/max(m,n). */ \
/*
				PASTEMAC(ch,scalv) \
				( \
				  BLIS_NO_CONJUGATE, \
				  n_elem - 1, \
				  &beta, \
				  x2, incx, \
				  cntx  \
				); \
*/ \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( randm_unb_var1,  randv_unb_var1 )
INSERT_GENTFUNC_BASIC( randnm_unb_var1, randnv_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* scale, \
       ctype_r* sumsq, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype_r zero_r = *PASTEMAC(chr,0); \
	ctype_r one_r  = *PASTEMAC(chr,1); \
\
	ctype*  chi1; \
	ctype_r chi1_r; \
	ctype_r chi1_i; \
	ctype_r scale_r; \
	ctype_r sumsq_r; \
	ctype_r abs_chi1_r; \
	ctype_r abs_chi1_i; \
	dim_t   i; \
\
	/* NOTE: This function attempts to mimic the algorithm for computing
	   the Frobenius norm in netlib LAPACK's ?lassq(). */ \
\
	/* Copy scale and sumsq to local variables. */ \
	bli_tcopys( chr,chr, *scale, scale_r ); \
	bli_tcopys( chr,chr, *sumsq, sumsq_r ); \
\
	chi1 = x; \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		/* Get the real and imaginary components of chi1. */ \
		bli_tgets( ch,chr, *chi1, chi1_r, chi1_i ); \
\
		abs_chi1_r = bli_fabs( chi1_r ); \
		abs_chi1_i = bli_fabs( chi1_i ); \
\
		if ( PASTEMAC(chr,isnan)( abs_chi1_r ) ) \
		{ \
			sumsq_r = abs_chi1_r; \
			scale_r = one_r; \
		} \
\
		if ( PASTEMAC(chr,isnan)( abs_chi1_i ) ) \
		{ \
			sumsq_r = abs_chi1_i; \
			scale_r = one_r; \
		} \
\
		if ( PASTEMAC(chr,isnan)( sumsq_r ) ) \
		{ \
			chi1 += incx; \
			continue; \
		} \
\
		if ( PASTEMAC(chr,isinf)( abs_chi1_r ) ) \
		{ \
			sumsq_r = abs_chi1_r; \
			scale_r = one_r; \
		} \
\
		if ( PASTEMAC(chr,isinf)( abs_chi1_i ) ) \
		{ \
			sumsq_r = abs_chi1_i; \
			scale_r = one_r; \
		} \
\
		if ( PASTEMAC(chr,isinf)( sumsq_r ) ) \
		{ \
			chi1 += incx; \
			continue; \
		} \
\
		/* Accumulate real component into sumsq, adjusting scale if
		   needed. */ \
		if ( abs_chi1_r > zero_r ) \
		{ \
			if ( scale_r < abs_chi1_r ) \
			{ \
				sumsq_r = one_r + \
				          sumsq_r * ( scale_r / abs_chi1_r ) * \
				                    ( scale_r / abs_chi1_r );  \
\
				bli_tcopys( chr,chr, abs_chi1_r, scale_r ); \
			} \
			else \
			{ \
				sumsq_r = sumsq_r + ( abs_chi1_r / scale_r ) * \
				                    ( abs_chi1_r / scale_r );  \
			} \
		} \
\
		/* Accumulate imaginary component into sumsq, adjusting scale if
		   needed. */ \
		if ( abs_chi1_i > zero_r ) \
		{ \
			if ( scale_r < abs_chi1_i ) \
			{ \
				sumsq_r = one_r + \
				          sumsq_r * ( scale_r / abs_chi1_i ) * \
				                    ( scale_r / abs_chi1_i );  \
\
				bli_tcopys( chr,chr, abs_chi1_i, scale_r ); \
			} \
			else \
			{ \
				sumsq_r = sumsq_r + ( abs_chi1_i / scale_r ) * \
				                    ( abs_chi1_i / scale_r );  \
			} \
		} \
\
		chi1 += incx; \
	} \
\
	/* Store final values of scale and sumsq to output variables. */ \
	bli_tcopys( chr,chr, scale_r, *scale ); \
	bli_tcopys( chr,chr, sumsq_r, *sumsq ); \
}

INSERT_GENTFUNCR_BASIC( sumsqv_unb_var1 )

// -----------------------------------------------------------------------------

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
bool PASTEMAC(ch,opname) \
     ( \
       conj_t conjx, \
       dim_t  n, \
       ctype* x, inc_t incx, \
       ctype* y, inc_t incy  \
     ) \
{ \
	for ( dim_t i = 0; i < n; ++i ) \
	{ \
		ctype* chi1 = x + (i  )*incx; \
		ctype* psi1 = y + (i  )*incy; \
\
		ctype chi1c; \
\
		if ( bli_is_conj( conjx ) ) { bli_tcopyjs( ch,ch, *chi1, chi1c ); } \
		else                        { bli_tcopys( ch,ch, *chi1, chi1c ); } \
\
		if ( !PASTEMAC(t,eqs)( ch,ch,ch, chi1c, *psi1 ) ) \
			return FALSE; \
	} \
\
	return TRUE; \
}

INSERT_GENTFUNC_BASIC( eqv_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
bool PASTEMAC(ch,opname) \
     ( \
       doff_t  diagoffx, \
       diag_t  diagx, \
       uplo_t  uplox, \
       trans_t transx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       ctype*  y, inc_t rs_y, inc_t cs_y  \
     ) \
{ \
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
	/* In the odd case where we are comparing against a complete unstored
	   matrix, we assert equality. Why? We assume the matrices are equal
	   unless we can find two corresponding elements that are unequal. So
	   if there are no elements, there is no inequality. Granted, this logic
	   is strange to think about no matter what, and thankfully it should
	   never be used under normal usage. */ \
	if ( bli_is_zeros( uplox_eff ) ) return TRUE; \
\
	/* Extract the conjugation component from the transx parameter. */ \
	conjx = bli_extract_conj( transx ); \
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
			for ( dim_t i = 0; i < n_elem; ++i ) \
			{ \
				ctype* x11 = x1 + (i  )*incx; \
				ctype* y11 = y1 + (i  )*incy; \
				ctype  x11c; \
\
				if ( bli_is_conj( conjx ) ) { bli_tcopyjs( ch,ch, *x11, x11c ); } \
				else                        { bli_tcopys( ch,ch, *x11, x11c ); } \
\
				if ( !PASTEMAC(t,eqs)( ch,ch,ch, x11c, *y11 ) ) \
					return FALSE; \
			} \
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
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					ctype* x11 = x1 + (i  )*incx; \
					ctype* y11 = y1 + (i  )*incy; \
					ctype  x11c; \
\
					if ( bli_is_conj( conjx ) ) { bli_tcopyjs( ch,ch, *x11, x11c ); } \
					else                        { bli_tcopys( ch,ch, *x11, x11c ); } \
\
					if ( !PASTEMAC(t,eqs)( ch,ch,ch, x11c, *y11 ) ) \
						return FALSE; \
				} \
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
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					ctype* x11 = x1 + (i  )*incx; \
					ctype* y11 = y1 + (i  )*incy; \
					ctype  x11c; \
\
					if ( bli_is_conj( conjx ) ) { bli_tcopyjs( ch,ch, *x11, x11c ); } \
					else                        { bli_tcopys( ch,ch, *x11, x11c ); } \
\
					if ( !PASTEMAC(t,eqs)( ch,ch,ch, x11c, *y11 ) ) \
						return FALSE; \
				} \
			} \
		} \
	} \
\
	return TRUE; \
}

INSERT_GENTFUNC_BASIC( eqm_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             FILE*  file, \
       const char*  s1, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const char*  format, \
       const char*  s2  \
     ) \
{ \
	const char default_spec[32] = PASTEMAC(ch,formatspec)(); \
\
	if ( format == NULL ) format = default_spec; \
\
	const ctype*chi1 = x; \
\
	fprintf( file, "%s\n", s1 ); \
\
	for ( dim_t i = 0; i < n; ++i ) \
	{ \
		bli_tfprints( ch, file, format, *chi1 ); \
		fprintf( file, "\n" ); \
\
		chi1 += incx; \
	} \
\
	fprintf( file, "%s\n", s2 ); \
}

INSERT_GENTFUNC_BASIC_I( fprintv )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             FILE*  file, \
       const char*  s1, \
             dim_t  m, \
             dim_t  n, \
       const ctype* x, inc_t rs_x, inc_t cs_x, \
       const char*  format, \
       const char*  s2  \
     ) \
{ \
	const char default_spec[32] = PASTEMAC(ch,formatspec)(); \
\
	if ( format == NULL ) format = default_spec; \
\
	fprintf( file, "%s\n", s1 ); \
\
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			const ctype* chi1 = (( ctype* ) x) + i*rs_x + j*cs_x; \
\
			bli_tfprints( ch, file, format, *chi1 ); \
			fprintf( file, " " ); \
		} \
\
		fprintf( file, "\n" ); \
	} \
\
	fprintf( file, "%s\n", s2 ); \
	fflush( file ); \
}

INSERT_GENTFUNC_BASIC_I( fprintm )

