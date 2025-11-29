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
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, arch, suf ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
             dim_t   n, \
       const void*   x0, inc_t incx, \
             dim_t*  index, \
       const cntx_t* cntx  \
     ) \
{ \
	const ctype*   x         = x0; \
\
	const ctype_r* minus_one = PASTEMAC(chr,m1); \
	const dim_t*   zero_i    = PASTEMAC(i,0); \
\
	ctype_r  chi1_r; \
	ctype_r  chi1_i; \
	ctype_r  abs_chi1; \
	ctype_r  abs_chi1_max; \
	dim_t    i_max_l; \
\
	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */ \
	if ( bli_zero_dim1( n ) ) \
	{ \
		bli_tcopys( i,i, *zero_i, *index ); \
		return; \
	} \
\
	/* Initialize the index of the maximum absolute value to zero. */ \
	bli_tcopys( i,i, *zero_i, i_max_l ); \
\
	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */ \
	bli_tcopys( chr,chr, *minus_one, abs_chi1_max ); \
\
	if ( incx == 1 ) \
	{ \
		const ctype* restrict chi1 = x; \
\
		for ( dim_t i = 0; i < n; ++i ) \
		{ \
			/* Get the real and imaginary components of chi1. */ \
			bli_tgets( ch,chr, *chi1, chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			bli_tabval2s( chr,chr,chr, chi1_r, chi1_r ); \
			bli_tabval2s( chr,chr,chr, chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			bli_tset0s( chr, abs_chi1 ); \
			bli_tadds( chr,chr,chr, chi1_r, abs_chi1 ); \
			bli_tadds( chr,chr,chr, chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || ( PASTEMAC(chr,isnan)( abs_chi1 ) && !PASTEMAC(chr,isnan)( abs_chi1_max ) ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				i_max_l      = i; \
			} \
\
			chi1 += 1; \
		} \
	} \
	else \
	{ \
		for ( dim_t i = 0; i < n; ++i ) \
		{ \
			const ctype* restrict chi1 = x + (i  )*incx; \
\
			/* Get the real and imaginary components of chi1. */ \
			bli_tgets( ch,chr, *chi1, chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			bli_tabval2s( chr,chr,chr, chi1_r, chi1_r ); \
			bli_tabval2s( chr,chr,chr, chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			bli_tset0s( chr, abs_chi1 ); \
			bli_tadds( chr,chr,chr, chi1_r, abs_chi1 ); \
			bli_tadds( chr,chr,chr, chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || ( PASTEMAC(chr,isnan)( abs_chi1 ) && !PASTEMAC(chr,isnan)( abs_chi1_max ) ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				i_max_l      = i; \
			} \
		} \
	} \
\
	/* Store the final index to the output variable. */ \
	bli_tcopys( i,i, i_max_l, *index ); \
}

INSERT_GENTFUNCR_BASIC( amaxv, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

