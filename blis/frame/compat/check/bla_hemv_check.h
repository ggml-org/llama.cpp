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

#if 1

#define bla_hemv_check( dt_str, op_str, uploa, m, lda, incx, incy ) \
{ \
	f77_int info = 0; \
	f77_int lower, upper; \
\
	lower = PASTEF77(lsame)( uploa, "L", (ftnlen)1, (ftnlen)1 ); \
	upper = PASTEF77(lsame)( uploa, "U", (ftnlen)1, (ftnlen)1 ); \
\
	if      ( !lower && !upper ) \
		info = 1; \
	else if ( *m < 0 ) \
		info = 2; \
	else if ( *lda < bli_max( 1, *m ) ) \
		info = 5; \
	else if ( *incx == 0 ) \
		info = 7; \
	else if ( *incy == 0 ) \
		info = 10; \
\
	if ( info != 0 ) \
	{ \
		char func_str[ BLIS_MAX_BLAS_FUNC_STR_LENGTH ]; \
\
		sprintf( func_str, "%s%-5s", dt_str, op_str ); \
\
		bli_string_mkupper( func_str ); \
\
		PASTEF77(xerbla)( func_str, &info, (ftnlen)6 ); \
\
		return; \
	} \
}

#endif
