/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022, Advanced Micro Devices, Inc.

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
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    x, const f77_int* incx, \
       const ftype*    beta, \
             ftype*    y, const f77_int* incy  \
     ) \
{ \
	trans_t blis_transa; \
	dim_t   m0, n0; \
	dim_t   m_y, n_x; \
	ftype*  x0; \
	ftype*  y0; \
	inc_t   incx0; \
	inc_t   incy0; \
\
	/* Initialize BLIS. */ \
	/*bli_init_auto();*/ \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  transa, \
	  m, \
	  n, \
	  lda, \
	  incx, \
	  incy  \
	); \
\
	/* BLAS handles cases where y has no elements as well as those where x has
	   no elements. In the case of the former, it cannot do any work since
	   the output vector is empty; but in the latter case, BLAS has peculiar
	   semantics. When x has no elements (and transa(A) has no columns), BLAS
	   returns immediately without performing any computation even if the
	   number of elements of y (and rows of transa(A)) is non-zero, in which
	   case any sane interpretations of gemv would have the the operation
	   reduce to y := beta * y. Here, we emulate the BLAS exactly so as to
	   provide "bug-for-bug" compatibility. Note that this extreme level of
	   compatibility would not be contemplated if it weren't for the fact
	   that some BLAS unit tests actually check for this behavior. Also, it
	   should be emphasized that BLIS, when called natively, does NOT exhibit
	   this quirky behavior; it will scale y by beta as one would expect. */ \
	if ( *m == 0 || *n == 0 ) \
	{ \
		/* Finalize BLIS. */ \
		/*bli_finalize_auto();*/ \
\
		return; \
	} \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
\
	/* Convert/typecast negative values of m and n to zero. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* Determine the dimensions of x and y so we can adjust the increments,
	   if necessary.*/ \
	bli_set_dims_with_trans( blis_transa, m0, n0, &m_y, &n_x ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n_x, (ftype*)x, *incx, x0, incx0 ); \
	bli_convert_blas_incv( m_y, (ftype*)y, *incy, y0, incy0 ); \
\
	/* If alpha is zero, scale y by beta and return early. */ \
	if ( PASTEMAC(ch,eq0)( *alpha ) ) \
	{ \
		PASTEMAC(ch,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m_y, \
		  ( ftype* )beta, \
		  ( ftype* )y0, incy0, \
		  NULL, \
		  NULL  \
		); \
		return; \
	} \
\
	/* Set the row and column strides of A. */ \
	const inc_t rs_a = 1; \
	const inc_t cs_a = *lda; \
\
	/* Declare a void function pointer for the current operation. */ \
	PASTECH(ch,blisname,_unb_ft) f; \
\
	/* Choose the underlying implementation. */ \
	if         ( bli_does_notrans( blis_transa ) )  f = PASTEMAC(ch,gemv_unf_var2); \
	else /* if ( bli_does_trans( blis_transa ) ) */ f = PASTEMAC(ch,gemv_unf_var1); \
\
	/* Obtain a valid context from the gks. This is needed because these
	   implementations of ?gemv_() skip calling gemv_ex() and instead
	   call the unblocked fused variants directly. */ \
	cntx_t* cntx = bli_gks_query_cntx(); \
\
	/* Invoke the variant chosen above, which loops over a level-1v or
	   level-1f kernel to implement the current operation. */ \
	f \
	( \
	  blis_transa, \
	  BLIS_NO_CONJUGATE, \
	  m0, \
	  n0, \
	  (ftype*)alpha, \
	  (ftype*)a, rs_a, cs_a, \
	  x0, incx0, \
	  (ftype*)beta, \
	  y0, incy0, \
	  cntx  \
	); \
\
	/* Finalize BLIS. */ \
	/*bli_finalize_auto();*/ \
}

#ifdef BLIS_ENABLE_BLAS
INSERT_GENTFUNC_BLAS( gemv, gemv )
#endif

