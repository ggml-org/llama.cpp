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
 * copys
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopys( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copys )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopyjs( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copyjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto x0 = convert<ctypex>( conj( x ) ); \
\
		INFO( "x:        " << x ); \
\
		bli_tcopyjs( chx,chx, x, x ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypex>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, copyjs_inplace )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopyris( chx,chy, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copyris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopyjris( chx,chy, \
		               real( x ), imag( x ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copyjris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( x ); \
		auto yir0 = convert<ctypey>( swapri( conj( x ) ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey yri, yir; \
		bli_tcopy1es( chx,chy, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypey>( yri, yri0 ); \
		check<ctypey>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, copy1es )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( conj( x ) ); \
		auto yir0 = convert<ctypey>( swapri( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey yri, yir; \
		bli_tcopyj1es( chx,chy, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypey>( yri, yri0 ); \
		check<ctypey>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, copyj1es )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopy1rs( chx,chy, x, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, copy1rs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopyj1rs( chx,chy, x, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, copyj1rs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypey,BLIS_NO_TRANSPOSE>( 1.0, xmn, 0.0, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
\
		bli_tcopys_mxn( chx,chy, M, N, &xmn[0][0], N, 1, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
\
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypey,BLIS_TRANSPOSE>( 1.0, xmn, 0.0, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
\
		bli_tcopys_mxn( chx,chy, N, M, &xmn[0][0], 1, N, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, copys_mxn )
