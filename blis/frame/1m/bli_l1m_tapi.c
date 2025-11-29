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
#define GENTFUNC( ctype, ch, opname, auxker ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Invoke the helper variant, which loops over the appropriate kernel
	   to implement the current operation. */ \
	PASTEMAC(ch,opname,_unb_var1) \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  transx, \
	  m, \
	  n, \
	  ( ctype* )x, rs_x, cs_x, \
	            y, rs_y, cs_y, \
	  ( cntx_t* )cntx  \
	); \
\
	/* When the diagonal of an upper- or lower-stored matrix is unit,
	   we handle it with a separate post-processing step. */ \
	/* NOTE: This code was disabled after I realized that when matrix A has the
	   properties of having a unit diagonal (and being lower or upper stored),
	   the operation should only read the strictly lower/upper triangle and
	   leave the diagonal of B untouched. */ \
/*
	if ( bli_is_upper_or_lower( uplox ) && \
	     bli_is_unit_diag( diagx ) ) \
	{ \
		PASTEMAC(ch,auxker,BLIS_TAPI_EX_SUF) \
		( \
		  diagoffx, \
		  diagx, \
		  transx, \
		  m, \
		  n, \
		  x, rs_x, cs_x, \
		  y, rs_y, cs_y, \
		  cntx, \
		  rntm  \
		); \
	} \
*/ \
}

INSERT_GENTFUNC_BASIC( addm, addd )
INSERT_GENTFUNC_BASIC( subm, subd )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Invoke the helper variant, which loops over the appropriate kernel
	   to implement the current operation. */ \
	PASTEMAC(ch,opname,_unb_var1) \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  transx, \
	  m, \
	  n, \
	  ( ctype* )x, rs_x, cs_x, \
	            y, rs_y, cs_y, \
	  ( cntx_t* )cntx  \
	); \
\
	/* When the diagonal of an upper- or lower-stored matrix is unit,
	   we handle it with a separate post-processing step. */ \
	/* NOTE: This code was disabled after I realized that when matrix A has the
	   properties of having a unit diagonal (and being lower or upper stored),
	   the operation should only read the strictly lower/upper triangle and
	   leave the diagonal of B untouched. */ \
/*
	if ( bli_is_upper_or_lower( uplox ) && \
	     bli_is_unit_diag( diagx ) ) \
	{ \
		doff_t diagoffy = diagoffx; \
		ctype* one      = PASTEMAC(ch,1); \
\
		if ( bli_does_trans( transx ) ) \
			bli_negate_diag_offset( &diagoffy ); \
\
		PASTEMAC(ch,setd,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  diagoffy, \
		  m, \
		  n, \
		  one, \
		  y, rs_y, cs_y, \
		  cntx, \
		  rntm  \
		); \
	} \
*/ \
}

INSERT_GENTFUNC_BASIC( copym )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If alpha is zero, then the entire operation is a no-op. */ \
	if ( bli_teq0s( ch, *alpha ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Invoke the helper variant, which loops over the appropriate kernel
	   to implement the current operation. */ \
	PASTEMAC(ch,opname,_unb_var1) \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  transx, \
	  m, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, rs_x, cs_x, \
	            y, rs_y, cs_y, \
	  ( cntx_t* )cntx  \
	); \
\
	/* When the diagonal of an upper- or lower-stored matrix is unit,
	   we handle it with a separate post-processing step. */ \
	/* NOTE: This code was disabled after I realized that when matrix A has the
	   properties of having a unit diagonal (and being lower or upper stored),
	   the operation should only read the strictly lower/upper triangle and
	   leave the diagonal of B untouched. */ \
/*
	if ( bli_is_upper_or_lower( uplox ) && \
	     bli_is_unit_diag( diagx ) ) \
	{ \
		PASTEMAC(ch,axpyd,BLIS_TAPI_EX_SUF) \
		( \
		  diagoffx, \
		  diagx, \
		  transx, \
		  m, \
		  n, \
		  alpha, \
		  x, rs_x, cs_x, \
		  y, rs_y, cs_y, \
		  cntx, \
		  rntm  \
		); \
	} \
*/ \
}

INSERT_GENTFUNC_BASIC( axpym )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* If alpha is zero, then we set the output matrix to zero. This
	   seemingly minor optimization is important because it will clear
	   any NaNs and Infs in x that would otherwise propogate. */ \
	if ( bli_teq0s( ch, *alpha ) ) \
	{ \
\
		PASTEMAC(ch,setm,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  diagoffx, \
		  diagx, \
		  uplox, \
		  m, \
		  n, \
		  alpha, \
		  y, rs_y, cs_y, \
		  cntx, \
		  rntm  \
		); \
		return; \
	} \
\
	/* Invoke the helper variant, which loops over the appropriate kernel
	   to implement the current operation. */ \
	PASTEMAC(ch,opname,_unb_var1) \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  transx, \
	  m, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, rs_x, cs_x, \
	            y, rs_y, cs_y, \
	  ( cntx_t* )cntx  \
	); \
\
	/* When the diagonal of an upper- or lower-stored matrix is unit,
	   we handle it with a separate post-processing step. */ \
	/* NOTE: This code was disabled after I realized that when matrix A has the
	   properties of having a unit diagonal (and being lower or upper stored),
	   the operation should only read the strictly lower/upper triangle and
	   leave the diagonal of B untouched. */ \
/*
	if ( bli_is_upper_or_lower( uplox ) && \
	     bli_is_unit_diag( diagx ) ) \
	{ \
		doff_t diagoffy = diagoffx; \
\
		if ( bli_does_trans( transx ) ) \
			bli_negate_diag_offset( &diagoffy ); \
\
		PASTEMAC(ch,setd,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  diagoffy, \
		  m, \
		  n, \
		  alpha, \
		  y, rs_y, cs_y, \
		  cntx, \
		  rntm  \
		); \
	} \
*/ \
}

INSERT_GENTFUNC_BASIC( scal2m )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjalpha, \
             doff_t diagoffx, \
             diag_t diagx, \
             uplo_t uplox, \
             dim_t  m, \
             dim_t  n, \
       const ctype* alpha, \
             ctype* x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Invoke the helper variant, which loops over the appropriate kernel
	   to implement the current operation. */ \
	PASTEMAC(ch,opname,_unb_var1) \
	( \
	  conjalpha, \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  m, \
	  n, \
	  ( ctype* )alpha, \
	            x, rs_x, cs_x, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( invscalm )
INSERT_GENTFUNC_BASIC( scalm )
INSERT_GENTFUNC_BASIC( setm )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
       const ctype*  beta, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* If beta is zero, then the operation reduces to copym. */ \
	if ( bli_teq0s( ch, *beta ) ) \
	{ \
		PASTEMAC(ch,copym,_unb_var1) \
		( \
		  diagoffx, \
		  diagx, \
		  uplox, \
		  transx, \
		  m, \
		  n, \
		  ( ctype* )x, rs_x, cs_x, \
		            y, rs_y, cs_y, \
		  ( cntx_t* )cntx  \
		); \
\
		return; \
	} \
\
	/* Invoke the helper variant, which loops over the appropriate kernel
	   to implement the current operation. */ \
	PASTEMAC(ch,opname,_unb_var1) \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  transx, \
	  m, \
	  n, \
	  ( ctype* )x, rs_x, cs_x, \
	  ( ctype* )beta, \
	            y, rs_y, cs_y, \
	  ( cntx_t* )cntx  \
	); \
\
	/* When the diagonal of an upper- or lower-stored matrix is unit,
	   we handle it with a separate post-processing step. */ \
	/* NOTE: This code was disabled after I realized that when matrix A has the
	   properties of having a unit diagonal (and being lower or upper stored),
	   the operation should only read the strictly lower/upper triangle and
	   leave the diagonal of B untouched. */ \
/*
	if ( bli_is_upper_or_lower( uplox ) && \
	     bli_is_unit_diag( diagx ) ) \
	{ \
		PASTEMAC(ch,xpbyd,BLIS_TAPI_EX_SUF) \
		( \
		  diagoffx, \
		  diagx, \
		  transx, \
		  m, \
		  n, \
		  x, rs_x, cs_x, \
		  beta, \
		  y, rs_y, cs_y, \
		  cntx, \
		  rntm  \
		); \
	} \
*/ \
}

INSERT_GENTFUNC_BASIC( xpbym )


#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_y, chx, chy, opname ) \
\
void PASTEMAC(chx,chy,opname,EX_SUF) \
     ( \
             doff_t   diagoffx, \
             diag_t   diagx, \
             uplo_t   uplox, \
             trans_t  transx, \
             dim_t    m, \
             dim_t    n, \
       const ctype_x* x, inc_t rs_x, inc_t cs_x, \
       const ctype_y* beta, \
             ctype_y* y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* If beta is zero, then the operation reduces to copym. */ \
	if ( bli_teq0s( chy, *beta ) ) \
	{ \
		PASTEMAC(chx,chy,castm) \
		( \
		  transx, \
		  m, \
		  n, \
		  x, rs_x, cs_x, \
		  y, rs_y, cs_y  \
		); \
\
		return; \
	} \
\
	/* Invoke the helper variant, which loops over the appropriate kernel
	   to implement the current operation. */ \
	PASTEMAC(chx,chy,opname,_unb_var1) \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  transx, \
	  m, \
	  n, \
	  ( ctype_x* )x, rs_x, cs_x, \
	  ( ctype_y* )beta, \
	              y, rs_y, cs_y, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC2_BASIC( xpbym_md )
INSERT_GENTFUNC2_MIX_DP( xpbym_md )


#endif

