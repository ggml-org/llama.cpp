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
 * scal2s
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2s( cha,chx,chy,chc, a, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, scal2s )

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
		INFO( "x:        " << x ); \
\
		bli_tscal2s( cha,chx,chx,chc, a, x, x ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, scal2s_inplace )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( conj( x ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2js( cha,chx,chy,chc, a, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, scal2js )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2ris( cha,chx,chy,chc, \
		               real( a ), imag( a ), \
		               real( x ), imag( x ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, scal2ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( conj( x ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2jris( cha,chx,chy,chc, \
		                real( a ), imag( a ), \
		                real( x ), imag( x ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, scal2jris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                             convert_prec<ctypec>( x ) ); \
		auto yir0 = swapri( conj( yri0 ) ); \
\
		INFO( "a:          " << a ); \
		INFO( "x:          " << x ); \
\
		ctypey yri, yir; \
		bli_tscal21es( cha,chx,chy,chc, a, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypec>( yri, yri0 ); \
		check<ctypec>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, R, scal21es )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                             convert_prec<ctypec>( conj( x ) ) ); \
		auto yir0 = swapri( conj( yri0 ) ); \
\
		INFO( "a:          " << a ); \
		INFO( "x:          " << x ); \
\
		ctypey yri, yir; \
		bli_tscal2j1es( cha,chx,chy,chc, a, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypec>( yri, yri0 ); \
		check<ctypec>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, R, scal2j1es )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal21rs( cha,chx,chy,chc, a, x, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, R, scal21rs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( conj( x ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2j1rs( cha,chx,chy,chc, a, x, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, R, scal2j1rs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
GENTFUNC0( opname, 1, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
GENTFUNC0( opname, 2, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
GENTFUNC0( opname, 5, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc )

#undef GENTFUNC0
#define GENTFUNC0( opname, D, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,PASTECH(opname,_,D)) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn00 = tile<M,N,ctypey>(); \
		      auto ymn = tile<M,D*N,ctypey>(); \
\
		INFO("column-major"); \
\
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, bli_is_conj( conjx ) ? conj( xmn ) : xmn, 0.0, ymn00, dense ); \
		auto ymn0 = bcast<D>( ymn00 ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2bbs_mxn( cha,chx,chy,chc, conjx, N, M, &a, &xmn[0][0], 1, N, &ymn[0][0], D, D*N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, scal2bbs_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, bli_is_conj( conjx ) ? conj( xmn ) : xmn, 0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2s_mxn( cha,chx,chy,chc, conjx, M, N, &a, &xmn[0][0], N, 1, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N,ctypey>(); \
\
		INFO("column-major"); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, bli_is_conj( conjx ) ? conj( xmn ) : xmn, 0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2s_mxn( cha,chx,chy,chc, conjx, N, M, &a, &xmn[0][0], 1, N, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, scal2s_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                         ? conj( xmn ) \
		                                         : xmn, \
		                                      0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, conjx, \
		                   M, N, &a, \
		                   &xmn[0][0], N, 1, \
		                   &ymn[0][0], 2*N, 2, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                      ? conj( xmn ) \
		                                      : xmn, \
		                                   0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, \
		                   conjx, N, M, &a, \
		                   &xmn[0][0], 1, N, \
		                   &ymn[0][0], 2, 2*N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, R, scal2ris_mxn_together )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	using ctypeyr = make_real_t<ctypey>; \
\
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto yrmn = tile<M,N,ctypeyr>(); \
		      auto yimn = tile<M,N,ctypeyr>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = tile<M,N,ctypey>(); \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                         ? conj( xmn ) \
		                                         : xmn, \
		                                      0.0, ymn0, dense ); \
		auto yrmn0 = real( ymn0 ); \
		auto yimn0 = imag( ymn0 ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, \
		                   conjx, M, N, &a, \
		                   &xmn[0][0], N, 1, \
		                   &yrmn[0][0], N, 1, \
		                   &yimn[0][0] - &yrmn[0][0] ); \
\
		INFO( "yr (C++):\n" << yrmn0 ); \
		INFO( "yi (C++):\n" << yimn0 ); \
		INFO( "yr (BLIS):\n" << yrmn ); \
		INFO( "yi (BLIS):\n" << yimn ); \
\
		check<ctypec>( yrmn, yrmn0 ); \
		check<ctypec>( yimn, yimn0 ); \
	} \
\
	for ( const auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	{ \
		const auto xmn = tile<M,N>( x ); \
		      auto yrmn = tile<M,N,ctypeyr>(); \
		      auto yimn = tile<M,N,ctypeyr>(); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M,N,ctypey>(); \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                      ? conj( xmn ) \
		                                      : xmn, \
		                                   0.0, ymn0, dense ); \
		auto yrmn0 = real( ymn0 ); \
		auto yimn0 = imag( ymn0 ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, \
		                   conjx, N, M, &a, \
		                   &xmn[0][0], 1, N, \
		                   &yrmn[0][0], 1, N, \
		                   &yimn[0][0] - &yrmn[0][0] ); \
\
		INFO( "yr (C++):\n" << yrmn0 ); \
		INFO( "yi (C++):\n" << yimn0 ); \
		INFO( "yr (BLIS):\n" << yrmn ); \
		INFO( "yi (BLIS):\n" << yimn ); \
\
		check<ctypec>( yrmn, yrmn0 ); \
		check<ctypec>( yimn, yimn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, R, scal2ris_mxn_separate )
