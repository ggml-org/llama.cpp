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
 * inverts
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinverts( chx,chc, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, R, inverts )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinvertris( chx,chc, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, R, invertris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto xri = x; \
		auto xir = swapri( conj( x ) ); \
\
		auto xri0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                             convert_prec<ctypec>( x ) ); \
		auto xir0 = swapri( conj( xri0 ) ); \
\
		INFO( "x:          " << x ); \
		INFO( "xri (orig): " << xri ); \
		INFO( "xir (orig): " << xir ); \
\
		bli_tinvert1es( chx,chc, xri, xir ); \
\
		INFO( "xri (C++):  " << xri0 ); \
		INFO( "xir (C++):  " << xir0 ); \
		INFO( "xri (BLIS): " << xri ); \
		INFO( "xir (BLIS): " << xir ); \
\
		check<ctypec>( xri, xri0 ); \
		check<ctypec>( xir, xir0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, R, invert1es )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto x0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
\
		bli_tinvert1rs( chx,chc, real( x ), imag( x ) ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, R, invert1rs )
