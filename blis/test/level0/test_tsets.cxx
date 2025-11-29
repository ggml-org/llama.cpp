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
 * sets
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
		bli_tsets( chx,chy, real( x ), imag( x ), y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, sets )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( real( x ) ); \
\
		INFO( "x:        " << x ); \
\
		bli_tsetrs( chx,chy, real( x ), y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, setrs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		imag( y0 ) = convert_prec<ctypey>( imag( x ) ); \
\
		INFO( "x:        " << x ); \
\
		bli_tsetis( chx,chy, imag( x ), y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, setis )

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
		bli_tsetris( chx,chy, \
		             real( x ), imag( x ), \
		             real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, setris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( 0.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tset0s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( 1.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tset1s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set1s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( 0.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tsetr0s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, setr0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		imag( y0 ) = convert_prec<ctypey>( 0.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tseti0s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, seti0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( 0.0 ); \
\
		bli_tset0ris( chy, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
  \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = tile<M,N>( convert<ctypey>( 0.0 ) ); \
\
		INFO( "y (init):\n" << ymn); \
\
		bli_tset0s_mxn( chy, M, N, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
  \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M,N>( convert<ctypey>( 0.0 ) ); \
\
		INFO( "y (init):\n" << ymn ); \
\
		bli_tset0s_mxn( chy, N, M, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0s_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
GENTFUNC0( opname, 1, ctypey, chy ) \
GENTFUNC0( opname, 2, ctypey, chy ) \
GENTFUNC0( opname, 5, ctypey, chy )

#undef GENTFUNC0
#define GENTFUNC0( opname, D, ctypey, chy ) \
UNIT_TEST(chy,PASTECH(opname,_,D)) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
  \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M,D*N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M,D*N>( convert<ctypey>( 0.0 ) ); \
\
		INFO( "y (init):\n" << ymn ); \
\
		bli_tset0bbs_mxn( chy, N, M, &ymn[0][0], D, D*N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0bbs_mxn )

#undef GENTFUNC0
#define GENTFUNC0( opname, D, ctypey, chy ) \
UNIT_TEST(chy,PASTECH(opname,_,D)) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
  \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M,D*N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M,D*N>( y ); \
\
		if constexpr ( is_complex<ctypey>::value ) \
		{ \
			for ( auto i = 0;i < M;i++ ) \
			for ( auto j = 0;j < N;j++ ) \
			{ \
				auto ymnij = &real( ymn0[i][j*D] ); \
				for ( auto d = 0;d < D;d++ ) \
				{ \
					ymnij[  d] = real( y ); \
					ymnij[D+d] = imag( y ); \
				} \
			} \
		} \
\
		for ( auto i = 0;i < M;i++ ) \
		for ( auto j = 0;j < N;j++ ) \
		for ( auto d = 1;d < D;d++ ) \
			ymn[i][j*D + d] = ctypey{}; \
\
		INFO( "y (init):\n" << ymn ); \
\
		bli_tbcastbbs_mxn( chy, N, M, &ymn[0][0], D, D*N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, bcastbbs_mxn )

#undef GENTFUNC0
#define GENTFUNC0( opname, D, ctypey, chy ) \
UNIT_TEST(chy,PASTECH(opname,_,D)) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
  \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M,D*N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M,D*N>( y ); \
\
		INFO( "y (init):\n" << ymn ); \
\
		bli_tbcastbbs_mxn( chy, N, M, &ymn[0][0], D, D*N ); \
		bli_tcompressbbs_mxn( chy, N, M, &ymn[0][0], D, D*N ); \
\
		for ( auto i = 0;i < M;i++ ) \
		for ( auto j = 0;j < N;j++ ) \
		for ( auto d = 1;d < D;d++ ) \
			ymn[i][j*D+d] = ymn0[i][j*D+d]; \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, compressbbs_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
GENTFUNC0( opname, 10, 10, ctypey, chy ) \
GENTFUNC0( opname, 10, 4, ctypey, chy ) \
GENTFUNC0( opname, 4, 10, ctypey, chy ) \
GENTFUNC0( opname, 10, 0, ctypey, chy ) \
GENTFUNC0( opname, 0, 10, ctypey, chy ) \
GENTFUNC0( opname, 4, 0, ctypey, chy ) \
GENTFUNC0( opname, 0, 4, ctypey, chy ) \
GENTFUNC0( opname, 0, 0, ctypey, chy )

#undef GENTFUNC0
#define GENTFUNC0( opname, M, N, ctypey, chy ) \
UNIT_TEST(chy,PASTECH(opname,_,M,_,N)) \
( \
	constexpr auto M0 = 10; \
	constexpr auto N0 = 10; \
  \
	for ( const auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M0,N0>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M0,N0>( convert<ctypey>( 0.0 ) ); \
		for ( auto i = 0; i < M; i++ ) \
		for ( auto j = 0; j < N; j++ ) \
			ymn0[i][j] = y; \
\
		INFO( "y (init):\n" << ymn ); \
\
		bli_tset0s_edge( chy, N, N0, M, M0, &ymn[0][0], N0 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0s_edge )
