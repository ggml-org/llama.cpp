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
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
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
	const num_t dt = PASTEMAC(ch,type); \
\
	const ctype* x1; \
	ctype*       y1; \
	conj_t       conjx; \
	dim_t        n_elem; \
	dim_t        offx, offy; \
	inc_t        incx, incy; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, transx, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_2d \
	( \
	  diagoffx, transx, \
	  m, n, rs_x, cs_x, rs_y, cs_y, \
	  &offx, &offy, &n_elem, &incx, &incy \
	); \
\
	conjx = bli_extract_conj( transx ); \
\
	if ( bli_is_nonunit_diag( diagx ) ) \
	{ \
		x1   = x + offx; \
		y1   = y + offy; \
	} \
	else /* if ( bli_is_unit_diag( diagx ) ) */ \
	{ \
		/* Simulate a unit diagonal for x with a zero increment over a unit
		   scalar. */ \
		x1   = PASTEMAC(ch,1); \
		incx = 0; \
		y1   = y + offy; \
	} \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Query the context for the operation's kernel address. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the kernel with the appropriate parameters. */ \
	f \
	( \
	  conjx, \
	  n_elem, \
	  ( ctype* )x1, incx, \
	            y1, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( addd,  addv,  BLIS_ADDV_KER )
INSERT_GENTFUNC_BASIC( copyd, copyv, BLIS_COPYV_KER )
INSERT_GENTFUNC_BASIC( subd,  subv,  BLIS_SUBV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
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
	const num_t dt = PASTEMAC(ch,type); \
\
	const ctype* x1; \
	ctype*       y1; \
	conj_t       conjx; \
	dim_t        n_elem; \
	dim_t        offx, offy; \
	inc_t        incx, incy; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, transx, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_2d \
	( \
	  diagoffx, transx, \
	  m, n, rs_x, cs_x, rs_y, cs_y, \
	  &offx, &offy, &n_elem, &incx, &incy \
	); \
\
	conjx = bli_extract_conj( transx ); \
\
	if ( bli_is_nonunit_diag( diagx ) ) \
	{ \
		x1   = x + offx; \
		y1   = y + offy; \
	} \
	else /* if ( bli_is_unit_diag( diagx ) ) */ \
	{ \
		/* Simulate a unit diagonal for x with a zero increment over a unit
		   scalar. */ \
		x1   = PASTEMAC(ch,1); \
		incx = 0; \
		y1   = y + offy; \
	} \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Query the context for the operation's kernel address. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the kernel with the appropriate parameters. */ \
	f \
	( \
	  conjx, \
	  n_elem, \
	  ( ctype* )alpha, \
	  ( ctype* )x1, incx, \
	            y1, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( axpyd,  axpyv,  BLIS_AXPYV_KER )
INSERT_GENTFUNC_BASIC( scal2d, scal2v, BLIS_SCAL2V_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       doff_t diagoffx, \
       dim_t  m, \
       dim_t  n, \
       ctype* x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*      x1; \
	dim_t       n_elem; \
	dim_t       offx; \
	inc_t       incx; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, BLIS_NO_TRANSPOSE, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_1d \
	( \
	  diagoffx, \
	  m, n, rs_x, cs_x, \
	  &offx, &n_elem, &incx \
	); \
\
	x1 = x + offx; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Query the context for the operation's kernel address. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the kernel with the appropriate parameters. */ \
	f \
	( \
	  n_elem, \
	  x1, incx, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( invertd, invertv, BLIS_INVERTV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjalpha, \
             doff_t diagoffx, \
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
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*      x1; \
	dim_t       n_elem; \
	dim_t       offx; \
	inc_t       incx; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, BLIS_NO_TRANSPOSE, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_1d \
	( \
	  diagoffx, \
	  m, n, rs_x, cs_x, \
	  &offx, &n_elem, &incx \
	); \
\
	x1 = x + offx; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Query the context for the operation's kernel address. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the kernel with the appropriate parameters. */ \
	f \
	( \
	  conjalpha, \
	  n_elem, \
	  ( ctype* )alpha, \
	            x1, incx, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( invscald, invscalv, BLIS_INVSCALV_KER )
INSERT_GENTFUNC_BASIC( scald, scalv, BLIS_SCALV_KER )
INSERT_GENTFUNC_BASIC( setd,  setv,  BLIS_SETV_KER )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t   diagoffx, \
             dim_t    m, \
             dim_t    n, \
       const ctype_r* alpha, \
             ctype*   x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt   = PASTEMAC(ch,type); \
	const num_t dt_r = PASTEMAC(chr,type); \
\
	ctype_r*    x1; \
	dim_t       n_elem; \
	dim_t       offx; \
	inc_t       incx; \
\
	/* If the datatype is real, the entire operation is a no-op. */ \
	if ( bli_is_real( dt ) ) return; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, BLIS_NO_TRANSPOSE, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_1d \
	( \
	  diagoffx, \
	  m, n, rs_x, cs_x, \
	  &offx, &n_elem, &incx \
	); \
\
	/* Alternate implementation. (Substitute for remainder of function). */ \
	/* for ( i = 0; i < n_elem; ++i ) \
	{ \
		ctype* chi11 = x1 + (i  )*incx; \
\
		bli_tsetis( ch,ch, *alpha, *chi11 ); \
	} */ \
\
	/* Acquire the address of the imaginary component of the first element,
	   and scale the increment for use in the real domain. Note that the
	   indexing into the imaginary field only needs to work for complex
	   datatypes since we return early for real domain types. */ \
	x1   = ( ctype_r* )( x + offx ) + 1; \
	incx = 2*incx; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Query the context for the operation's kernel address. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt_r, kerid, cntx ); \
\
	/* Invoke the kernel with the appropriate parameters. */ \
	f \
	( \
	  BLIS_NO_CONJUGATE, \
	  n_elem, \
	  ( ctype_r* )alpha, \
	              x1, incx, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNCR_BASIC( setid, setv, BLIS_SETV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t diagoffx, \
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
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*      x1; \
	dim_t       n_elem; \
	dim_t       offx; \
	inc_t       incx; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, BLIS_NO_TRANSPOSE, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_1d \
	( \
	  diagoffx, \
	  m, n, rs_x, cs_x, \
	  &offx, &n_elem, &incx \
	); \
\
	x1 = x + offx; \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Query the context for the operation's kernel address. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the kernel with the appropriate parameters. */ \
	f \
	( \
	  BLIS_NO_CONJUGATE, \
	  n_elem, \
	  ( ctype* )alpha, 0, \
	            x1, incx, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( shiftd, addv, BLIS_ADDV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kername, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
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
	const num_t dt = PASTEMAC(ch,type); \
\
	const ctype* x1; \
	ctype*       y1; \
	conj_t       conjx; \
	dim_t        n_elem; \
	dim_t        offx, offy; \
	inc_t        incx, incy; \
\
	if ( bli_zero_dim2( m, n ) ) return; \
\
	if ( bli_is_outside_diag( diagoffx, transx, m, n ) ) return; \
\
	/* Determine the distance to the diagonals, the number of diagonal
	   elements, and the diagonal increments. */ \
	bli_set_dims_incs_2d \
	( \
	  diagoffx, transx, \
	  m, n, rs_x, cs_x, rs_y, cs_y, \
	  &offx, &offy, &n_elem, &incx, &incy \
	); \
\
	conjx = bli_extract_conj( transx ); \
\
	if ( bli_is_nonunit_diag( diagx ) ) \
	{ \
		x1   = x + offx; \
		y1   = y + offy; \
	} \
	else /* if ( bli_is_unit_diag( diagx ) ) */ \
	{ \
		/* Simulate a unit diagonal for x with a zero increment over a unit
		   scalar. */ \
		x1   = PASTEMAC(ch,1); \
		incx = 0; \
		y1   = y + offy; \
	} \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Query the context for the operation's kernel address. */ \
	PASTECH(kername,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the kernel with the appropriate parameters. */ \
	f \
	( \
	  conjx, \
	  n_elem, \
	  ( ctype* )x1, incx, \
	  ( ctype* )beta, \
	            y1, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( xpbyd,  xpbyv,  BLIS_XPBYV_KER )


#endif

