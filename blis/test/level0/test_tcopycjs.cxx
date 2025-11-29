/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Southern Methodist University

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "test_l0.hpp"

/******************************************************************************
 *
 * copycjs
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( bli_is_conj( conjx ) ? conj( x ) : x ); \
\
		INFO( "conjx:    " << bli_is_conj( conjx ) ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopycjs( chx,chy, conjx, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copycjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for (       auto x : test_values<ctypex>() ) \
	{ \
		auto x0 = convert<ctypex>( bli_is_conj( conjx ) ? conj( x ) : x ); \
\
		INFO( "conjx:    " << bli_is_conj( conjx ) ); \
		INFO( "x:        " << x ); \
\
		bli_tcopycjs( chx,chx, conjx, x, x ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypex>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, copycjs_inplace )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( bli_is_conj( conjx ) ? conj( x ) : x ); \
\
		INFO( "conjx:    " << bli_is_conj( conjx ) ); \
		INFO( "x:        " << x); \
\
		ctypey y; \
		bli_tcopycjris( chx,chy, conjx, \
		                real( x ), imag( x ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copycjris )
