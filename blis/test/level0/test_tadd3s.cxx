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
 * add3s
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypez, chz, ctypec, chc ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3s( chx,chy,chz,chc, x, y, z ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3s);

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		bli_tadd3s( chx,chy,chy,chc, x, y, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
\
	for (       auto x : test_values<ctypex>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto x0 = convert<ctypex>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		bli_tadd3s( chx,chy,chx,chc, x, y, x ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX3(RC, RC, C, add3s_inplace);

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypez, chz, ctypec, chc ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3js( chx,chy,chz,chc, x, y, z ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3js);

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypez, chz, ctypec, chc ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3ris( chx,chy,chz,chc, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ), \
		              real( z ), imag( z ) ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3ris);

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypez, chz, ctypec, chc ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3jris( chx,chy,chz,chc, \
		               real( x ), imag( x ), \
		               real( y ), imag( y ), \
		               real( z ), imag( z ) ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3jris);
