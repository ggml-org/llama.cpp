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
 * eqs
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto expected = convert_prec<ctypec>( x ) == \
		                convert_prec<ctypec>( y ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		auto found = bli_teqs( chx,chy,chc, x, y ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, eqs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto expected = convert_prec<ctypec>( x ) == \
		                convert_prec<ctypec>( y ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		auto found = bli_teqris( chx,chy,chc, \
		                         real(x), imag(x), \
		                         real(y), imag(y) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, eqris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq1ris( chx, real( x ), imag( x ) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq1ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 0.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq0ris( chx, real( x ), imag( x ) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq0ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( -1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teqm1ris( chx, real( x ), imag( x ) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eqm1ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq1s( chx, x ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq1s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 0.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq0s( chx, x ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( -1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teqm1s( chx, x ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eqm1s )
