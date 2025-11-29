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
 * axpbys
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto b : test_values<ctypeb>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( b ) * \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbys( cha,chx,chb,chy,chc, a, x, b, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, R, axpbys )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for (       auto x : test_values<ctypex>() ) \
	for ( const auto b : test_values<ctypeb>() ) \
	{ \
		auto x0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( b ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
\
		bli_taxpbys( cha,chx,chb,chx,chc, a, x, b, x ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, axpbys_inplace )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto b : test_values<ctypeb>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( a ) * \
		                           conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( b ) * \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbyjs( cha,chx,chb,chy,chc, a, x, b, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, R, axpbyjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto b : test_values<ctypeb>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( b ) * \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbyris( cha,chx,chb,chy,chc, \
		               real( a ), imag( a ), \
		               real( x ), imag( x ), \
		               real( b ), imag( b ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, R, axpbyris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto b : test_values<ctypeb>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( a ) * \
		                           conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( b ) * \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbyjris( cha,chx,chb,chy,chc, \
		                real( a ), imag( a ), \
		                real( x ), imag( x ), \
		                real( b ), imag( b ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, R, axpbyjris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto b : test_values<ctypeb>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N>( y ); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, xmn, b, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_taxpbys_mxn( cha,chx,chb,chy,chc, M, N, &a, &xmn[0][0], N, 1, &b, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto b : test_values<ctypeb>() ) \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, xmn, b, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_taxpbys_mxn( cha,chx,chb,chy,chc, N, M, &a, &xmn[0][0], 1, N, &b, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, R, axpbys_mxn )
