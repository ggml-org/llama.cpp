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

// Guard the function definitions so that they are only compiled when
// #included from files that define the typed API macros.
#ifdef BLIS_ENABLE_TAPI

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, ftname, rvarname, cvarname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             trans_t transa, \
             conj_t  conjx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  x, inc_t incx, \
       const ctype*  beta, \
             ctype*  y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	dim_t m_y, n_x; \
\
	/* Determine the dimensions of y and x. */ \
	bli_set_dims_with_trans( transa, m, n, &m_y, &n_x ); \
\
	/* If y has zero elements, return early. */ \
	if ( bli_zero_dim1( m_y ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* If x has zero elements, or if alpha is zero, scale y by beta and
	   return early. */ \
	if ( bli_zero_dim1( n_x ) || bli_teq0s( ch, *alpha ) ) \
	{ \
		PASTEMAC(ch,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m_y, \
		  beta, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
		return; \
	} \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,ftname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if ( bli_does_notrans( transa ) ) \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,rvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,cvarname); \
	} \
	else /* if ( bli_does_trans( transa ) ) */ \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,cvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,rvarname); \
	} \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  transa, \
	  conjx, \
	  m, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )a, rs_a, cs_a, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	  ( cntx_t* )cntx \
	); \
}

INSERT_GENTFUNC_BASIC( gemv, gemv, gemv_unf_var1, gemv_unf_var2 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, ftname, rvarname, cvarname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  m, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             ctype* a, inc_t rs_a, inc_t cs_a  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	/* If x or y has zero elements, or if alpha is zero, return early. */ \
	if ( bli_zero_dim2( m, n ) || bli_teq0s( ch, *alpha ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,ftname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,rvarname); \
	else /* column or general stored */    f = PASTEMAC(ch,cvarname); \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  conjx, \
	  conjy, \
	  m, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	  ( ctype* )y, incy, \
	            a, rs_a, cs_a, \
	  ( cntx_t* )cntx \
	); \
}

INSERT_GENTFUNC_BASIC( ger, ger, ger_unb_var1, ger_unb_var2 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, ftname, conjh, rvarname, cvarname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             uplo_t uploa, \
             conj_t conja, \
             conj_t conjx, \
             dim_t  m, \
       const ctype* alpha, \
       const ctype* a, inc_t rs_a, inc_t cs_a, \
       const ctype* x, inc_t incx, \
       const ctype* beta, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* If x has zero elements, or if alpha is zero, scale y by beta and
	   return early. */ \
	if ( bli_zero_dim1( m ) || bli_teq0s( ch, *alpha ) ) \
	{ \
		PASTEMAC(ch,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  beta, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
		return; \
	} \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,ftname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if ( bli_is_lower( uploa ) ) \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,rvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,cvarname); \
	} \
	else /* if ( bli_is_upper( uploa ) ) */ \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,cvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,rvarname); \
	} \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  uploa, \
	  conja, \
	  conjx, \
	  conjh, /* used by variants to distinguish hemv from symv */ \
	  m, \
	  ( ctype* )alpha, \
	  ( ctype* )a, rs_a, cs_a, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	  ( cntx_t* )cntx \
	); \
}

INSERT_GENTFUNC_BASIC( hemv, hemv, BLIS_CONJUGATE,    hemv_unf_var1, hemv_unf_var3 )
INSERT_GENTFUNC_BASIC( symv, hemv, BLIS_NO_CONJUGATE, hemv_unf_var1, hemv_unf_var3 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, ftname, conjh, rvarname, cvarname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             uplo_t   uploa, \
             conj_t   conjx, \
             dim_t    m, \
       const ctype_r* alpha, \
       const ctype*   x, inc_t incx, \
             ctype*   a, inc_t rs_a, inc_t cs_a  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	ctype alpha_local; \
\
	/* If x has zero elements, or if alpha is zero, return early. */ \
	if ( bli_zero_dim1( m ) || bli_teq0s( chr, *alpha ) ) return; \
\
	/* Make a local copy of alpha, cast into the complex domain. This
	   allows us to use the same underlying her variants to implement
	   both her and syr operations. */ \
	bli_tcopys( chr,ch, *alpha, alpha_local ); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,ftname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if ( bli_is_lower( uploa ) ) \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,rvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,cvarname); \
	} \
	else /* if ( bli_is_upper( uploa ) ) */ \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,cvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,rvarname); \
	} \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  uploa, \
	  conjx, \
	  conjh, /* used by variants to distinguish her from syr */ \
	  m, \
	  ( ctype* )&alpha_local, \
	  ( ctype* )x, incx, \
	            a, rs_a, cs_a, \
	  ( cntx_t* )cntx \
	); \
}

INSERT_GENTFUNCR_BASIC( her, her, BLIS_CONJUGATE, her_unb_var1, her_unb_var2 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, ftname, conjh, rvarname, cvarname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             uplo_t uploa, \
             conj_t conjx, \
             dim_t  m, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
             ctype* a, inc_t rs_a, inc_t cs_a  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	/* If x has zero elements, or if alpha is zero, return early. */ \
	if ( bli_zero_dim1( m ) || bli_teq0s( ch, *alpha ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,ftname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if ( bli_is_lower( uploa ) ) \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,rvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,cvarname); \
	} \
	else /* if ( bli_is_upper( uploa ) ) */ \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,cvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,rvarname); \
	} \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  uploa, \
	  conjx, \
	  conjh, /* used by variants to distinguish her2 from syr2 */ \
	  m, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	            a, rs_a, cs_a, \
	  ( cntx_t* )cntx \
	); \
}

INSERT_GENTFUNC_BASIC( syr, her, BLIS_NO_CONJUGATE, her_unb_var1, her_unb_var2 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, ftname, conjh, rvarname, cvarname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             uplo_t uploa, \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  m, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             ctype* a, inc_t rs_a, inc_t cs_a  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	/* If x has zero elements, or if alpha is zero, return early. */ \
	if ( bli_zero_dim1( m ) || bli_teq0s( ch, *alpha ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,ftname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if ( bli_is_lower( uploa ) ) \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,rvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,cvarname); \
	} \
	else /* if ( bli_is_upper( uploa ) ) */ \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,cvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,rvarname); \
	} \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  uploa, \
	  conjx, \
	  conjy, \
	  conjh, \
	  m, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	  ( ctype* )y, incy, \
	            a, rs_a, cs_a, \
	  ( cntx_t* )cntx \
	); \
}

INSERT_GENTFUNC_BASIC( her2, her2, BLIS_CONJUGATE,    her2_unf_var1, her2_unf_var4 )
INSERT_GENTFUNC_BASIC( syr2, her2, BLIS_NO_CONJUGATE, her2_unf_var1, her2_unf_var4 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, ftname, rvarname, cvarname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             uplo_t  uploa, \
             trans_t transa, \
             diag_t  diaga, \
             dim_t   m, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
             ctype*  x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	/* If x has zero elements, return early. */ \
	if ( bli_zero_dim1( m ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* If alpha is zero, set x to zero and return early. */ \
	if ( bli_teq0s( ch, *alpha ) ) \
	{ \
		PASTEMAC(ch,setv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  alpha, \
		  x, incx, \
		  cntx, \
		  NULL  \
		); \
		return; \
	} \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,ftname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if ( bli_does_notrans( transa ) ) \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,rvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,cvarname); \
	} \
	else /* if ( bli_does_trans( transa ) ) */ \
	{ \
		if ( bli_is_row_stored( rs_a, cs_a ) ) f = PASTEMAC(ch,cvarname); \
		else /* column or general stored */    f = PASTEMAC(ch,rvarname); \
	} \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  uploa, \
	  transa, \
	  diaga, \
	  m, \
	  ( ctype* )alpha, \
	  ( ctype* )a, rs_a, cs_a, \
	            x, incx, \
	  ( cntx_t* )cntx \
	); \
}

INSERT_GENTFUNC_BASIC( trmv, trmv, trmv_unf_var1, trmv_unf_var2 )
INSERT_GENTFUNC_BASIC( trsv, trmv, trsv_unf_var1, trsv_unf_var2 )


#endif

