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

#ifdef BLIS_ENABLE_BLAS

//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
ftype PASTEF77(ch,blasname,chc) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     ) \
{ \
	dim_t  n0; \
	ftype* x0; \
	ftype* y0; \
	inc_t  incx0; \
	inc_t  incy0; \
	ftype  rho; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Convert/typecast negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, (ftype*)x, *incx, x0, incx0 ); \
	bli_convert_blas_incv( n0, (ftype*)y, *incy, y0, incy0 ); \
\
	/* Call BLIS interface. */ \
	PASTEMAC(ch,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  blis_conjx, \
	  BLIS_NO_CONJUGATE, \
	  n0, \
	  x0, incx0, \
	  y0, incy0, \
	  &rho, \
	  NULL, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
\
	return rho; \
}

INSERT_GENTFUNCDOTR_BLAS( dot, dotv )

#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL

INSERT_GENTFUNCDOTC_BLAS( dot, dotv )

#else // #ifdef BLIS_ENABLE_COMPLEX_RETURN_INTEL

// For the "intel" complex return type, use a hidden preceding parameter to
// return the result rather than an actual return value.
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
void PASTEF77(ch,blasname,chc) \
     ( \
       ftype*         rhop, \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     ) \
{ \
	dim_t  n0; \
	ftype* x0; \
	ftype* y0; \
	inc_t  incx0; \
	inc_t  incy0; \
	ftype  rho; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Convert/typecast negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, (ftype*)x, *incx, x0, incx0 ); \
	bli_convert_blas_incv( n0, (ftype*)y, *incy, y0, incy0 ); \
\
	/* Call BLIS interface. */ \
	PASTEMAC(ch,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  blis_conjx, \
	  BLIS_NO_CONJUGATE, \
	  n0, \
	  x0, incx0, \
	  y0, incy0, \
	  &rho, \
	  NULL, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
\
	*rhop = rho; \
}

INSERT_GENTFUNCDOTC_BLAS( dot, dotv )

#endif


// -- "Black sheep" dot product function definitions --

// Input vectors stored in single precision, computed in double precision,
// with result returned in single precision.
float PASTEF77(sd,sdot)
     (
       const f77_int* n,
       const float*   sb,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
	return ( float )
	       (
	         ( double )(*sb) +
	         PASTEF77(d,sdot)
	         (
	           n,
	           x, incx,
	           y, incy
	         )
	       );
}

// Input vectors stored in single precision, computed in double precision,
// with result returned in double precision.
double PASTEF77(d,sdot)
     (
       const f77_int* n,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
	dim_t   n0;
	float*  x0;
	float*  y0;
	inc_t   incx0;
	inc_t   incy0;
	double  rho;
	dim_t   i;

	/* Initialization of BLIS is not required. */

	/* Convert/typecast negative values of n to zero. */
	bli_convert_blas_dim1( *n, n0 );

	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */
	bli_convert_blas_incv( n0, (float*)x, *incx, x0, incx0 );
	bli_convert_blas_incv( n0, (float*)y, *incy, y0, incy0 );

	rho = 0.0;

	for ( i = 0; i < n0; i++ )
	{
		float* chi1 = x0 + (i  )*incx0;
		float* psi1 = y0 + (i  )*incy0;

		bli_tdots( d,d,d,d, (( double )(*chi1)),
		                    (( double )(*psi1)), rho );
	}

	/* Finalization of BLIS is not required, because initialization was
	   not required. */

	return rho;
}

#endif

