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
             conj_t conjy, \
             dim_t  n, \
       const ctype* alphax, \
       const ctype* alphay, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             ctype* z, inc_t incz  \
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
	  ( ctype* )alphax, \
	  ( ctype* )alphay, \
	  ( ctype* )x, incx, \
	  ( ctype* )y, incy, \
	            z, incz, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( axpy2v, BLIS_AXPY2V_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conja, \
             conj_t conjx, \
             dim_t  m, \
             dim_t  b_n, \
       const ctype* alpha, \
       const ctype* a, inc_t inca, inc_t lda, \
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
	  conja, \
	  conjx, \
	  m, \
	  b_n, \
	  ( ctype* )alpha, \
	  ( ctype* )a, inca, lda, \
	  ( ctype* )x, incx, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( axpyf, BLIS_AXPYF_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjxt, \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             ctype* rho, \
             ctype* z, inc_t incz  \
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
	  conjxt, \
	  conjx, \
	  conjy, \
	  n, \
	  ( ctype* )alpha, \
	  ( ctype* )x, incx, \
	  ( ctype* )y, incy, \
	            rho, \
	            z, incz, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( dotaxpyv, BLIS_DOTAXPYV_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjat, \
             conj_t conja, \
             conj_t conjw, \
             conj_t conjx, \
             dim_t  m, \
             dim_t  b_n, \
       const ctype* alpha, \
       const ctype* a, inc_t inca, inc_t lda, \
       const ctype* w, inc_t incw, \
       const ctype* x, inc_t incx, \
       const ctype* beta, \
             ctype* y, inc_t incy, \
             ctype* z, inc_t incz  \
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
	  conjat, \
	  conja, \
	  conjw, \
	  conjx, \
	  m, \
	  b_n, \
	  ( ctype* )alpha, \
	  ( ctype* )a, inca, lda, \
	  ( ctype* )w, incw, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	            z, incz, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( dotxaxpyf, BLIS_DOTXAXPYF_KER )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, kerid ) \
\
void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjat, \
             conj_t conjx, \
             dim_t  m, \
             dim_t  b_n, \
       const ctype* alpha, \
       const ctype* a, inc_t inca, inc_t lda, \
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
	  conjat, \
	  conjx, \
	  m, \
	  b_n, \
	  ( ctype* )alpha, \
	  ( ctype* )a, inca, lda, \
	  ( ctype* )x, incx, \
	  ( ctype* )beta, \
	            y, incy, \
	  ( cntx_t* )cntx  \
	); \
}

INSERT_GENTFUNC_BASIC( dotxf, BLIS_DOTXF_KER )


#endif

