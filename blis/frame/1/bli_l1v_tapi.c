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
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  n, \
	  ( ctype* )x, incx, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( addv,  BLIS_ADDV_KER )
INSERT_GENTFUNC_BASIC( copyv, BLIS_COPYV_KER )
INSERT_GENTFUNC_BASIC( subv,  BLIS_SUBV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             dim_t  n, \
       const ctype* x, inc_t incx, \
             dim_t* index  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  n, \
	  ( ctype* )x, incx, \
	  index, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( amaxv, BLIS_AMAXV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* alpha, \
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
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( axpbyv, BLIS_AXPBYV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) \
		cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( axpyv,  BLIS_AXPYV_KER )
INSERT_GENTFUNC_BASIC( scal2v, BLIS_SCAL2V_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             ctype* rho  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  conjy, \
	  n, \
	  ( ctype* )x, incx, \
	  ( ctype* )y, incy, \
	            rho, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( dotv, BLIS_DOTV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
       const ctype* beta, \
             ctype* rho  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  conjy, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	  ( ctype* )y, incy, \
	  ( ctype* )beta, \
	            rho, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( dotxv, BLIS_DOTXV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       dim_t  n, \
       ctype* x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  n, \
	  x, incx, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( invertv, BLIS_INVERTV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjalpha, \
             dim_t  n, \
       const ctype* alpha, \
             ctype* x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjalpha, \
	  n, \
	  ( ctype* )alpha, \
	            x, incx, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( invscalv, BLIS_INVSCALV_KER )
INSERT_GENTFUNC_BASIC( scalv, BLIS_SCALV_KER )
INSERT_GENTFUNC_BASIC( setv,  BLIS_SETV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       dim_t  n, \
       ctype* x, inc_t incx, \
       ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_TAPI_EX_DECLS \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  n, \
	  x, incx, \
	  y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( swapv, BLIS_SWAPV_KER )

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
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
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Obtain a valid context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	PASTECH(opname,_ker_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	f \
	( \
	  conjx, \
	  n, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( xpbyv, BLIS_XPBYV_KER )


#endif

