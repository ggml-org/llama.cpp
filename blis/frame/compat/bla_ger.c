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
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjy, blasname, blisname ) \
\
void PASTEF77(ch,blasname,chc) \
     ( \
       const f77_int* m, \
       const f77_int* n, \
       const ftype*   alpha, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy, \
             ftype*   a, const f77_int* lda  \
     ) \
{ \
	dim_t   m0, n0; \
	ftype*  x0; \
	ftype*  y0; \
	inc_t   incx0; \
	inc_t   incy0; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  MKSTR(chc), \
	  m, \
	  n, \
	  incx, \
	  incy, \
	  lda  \
	); \
\
	/* Convert/typecast negative values of m and n to zero. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( m0, (ftype*)x, *incx, x0, incx0 ); \
	bli_convert_blas_incv( n0, (ftype*)y, *incy, y0, incy0 ); \
\
	/* Set the row and column strides of A. */ \
	const inc_t rs_a = 1; \
	const inc_t cs_a = *lda; \
\
	/* Call BLIS interface. */ \
	PASTEMAC(ch,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  BLIS_NO_CONJUGATE, \
	  blis_conjy, \
	  m0, \
	  n0, \
	  (ftype*)alpha, \
	  x0, incx0, \
	  y0, incy0, \
	  (ftype*)a,  rs_a, cs_a, \
	  NULL, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
}

#ifdef BLIS_ENABLE_BLAS
INSERT_GENTFUNCDOT_BLAS( ger, ger )
#endif

