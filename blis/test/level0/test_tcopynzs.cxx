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
 * copynzs
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( real( x ) ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = convert_prec<ctypey>( imag( x ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopynzs( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copynzs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( real( x ) ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = convert_prec<ctypey>( -imag( x ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopyjnzs( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copyjnzs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( real( x ) ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = convert_prec<ctypey>( imag( x ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopynzris( chx,chy, \
		                real( x ), imag( x ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copynzris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( real( x ) ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = convert_prec<ctypey>( -imag( x ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopyjnzris( chx,chy, \
		                 real( x ), imag( x ), \
		                 real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copyjnzris )
