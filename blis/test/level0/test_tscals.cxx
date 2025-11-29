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
 * scals
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscals( cha,chx,chc, a, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, scals )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( conj( a ) ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscaljs( cha,chx,chc, a, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, scaljs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscalris( cha,chx,chc, \
		              real( a ), imag( a ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, scalris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( conj( a ) ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscaljris( cha,chx,chc, \
		               real( a ), imag( a ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, scaljris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto xri = x; \
		auto xir = swapri( conj( x ) ); \
\
		auto xri0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                             convert_prec<ctypec>( x ) ); \
		auto xir0 = swapri( conj( xri0 ) ); \
\
		INFO( "a:          " << a ); \
		INFO( "x:          " << x ); \
		INFO( "xri (orig): " << xri ); \
		INFO( "xir (orig): " << xir ); \
\
		bli_tscal1es( cha,chx,chc, a, xri, xir ); \
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

INSERT_GENTFUNC_MIX3( RC, C, R, scal1es )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for (       auto x : test_values<ctypex>() ) \
	{ \
		auto x0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x (orig): " << x ); \
\
		bli_tscal1rs( cha,chx,chc, a, real( x ), imag( x ) ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "xr(BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, C, R, scal1rs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( const uplo_t uplo : { BLIS_UPPER, BLIS_LOWER } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto diagoff : { -1, 0, 1 } ) \
	{ \
		auto xmn = tile<M,N>( x ); \
\
		INFO( "row-major" ); \
\
		std::function<bool(size_t,size_t)> func = is_below( diagoff ); \
		if ( uplo == BLIS_UPPER ) func = is_above( diagoff ); \
\
		auto xmn0 = xmn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, xmn, 0.0, xmn0, func ); \
\
		INFO( "upper:   " << ( uplo == BLIS_UPPER ) ); \
		INFO( "diagoff: " << diagoff ); \
		INFO( "a: " << a ); \
		INFO( "x (init):\n" << xmn ); \
\
		bli_tscalris_mxn_uplo( cha,chx,chc, uplo, diagoff, M, N, \
		                       &real( a ), &real( a )+1, \
		                       &real( xmn[0][0] ), &real( xmn[0][0] )+1, \
		                       &real( xmn[1][0] ) - &real( xmn[0][0] ), \
		                       &real( xmn[0][1] ) - &real( xmn[0][0] ) ); \
\
		INFO( "x (C++):\n" << xmn0 ); \
		INFO( "x (BLIS):\n" << xmn ); \
\
		check<ctypec>( xmn, xmn0 ); \
	} \
\
	for ( const uplo_t uplo : { BLIS_UPPER, BLIS_LOWER } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for ( const auto diagoff : { -1, 0, 1 } ) \
	{ \
		auto xmn = tile<M,N>( x ); \
\
		INFO( "column-major" ); \
\
		std::function<bool(size_t,size_t)> func = is_below( diagoff ); \
		if ( uplo == BLIS_UPPER ) func = is_above( diagoff ); \
\
		auto xmn0 = xmn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, xmn, 0.0, xmn0, func ); \
\
		INFO( "upper:   " << ( uplo == BLIS_UPPER ) ); \
		INFO( "diagoff: " << diagoff ); \
		INFO( "a: " << a ); \
		INFO( "x (init):\n" << xmn ); \
\
		bli_tscalris_mxn_uplo( cha,chx,chc, uplo, diagoff, N, M, \
		                       &real( a ), &real( a )+1, \
		                       &real( xmn[0][0] ), &real( xmn[0][0] )+1, \
		                       &real( xmn[0][1] ) - &real( xmn[0][0] ), \
		                       &real( xmn[1][0] ) - &real( xmn[0][0] ) ); \
\
		INFO( "x (C++):\n" << xmn0 ); \
		INFO( "x (BLIS):\n" << xmn ); \
\
		check<ctypec>( xmn, xmn0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, scalris_mxn_uplo )
