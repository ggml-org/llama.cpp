/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#include "blis.h"

// Separate instantiation for ArmSVE reference kernels.
// Temporary workaround. Will be removed after upstream has switched to a better way
//  of exposing gemmsup interface.

//
// -- Row storage case ---------------------------------------------------------
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       conj_t              conja, \
       conj_t              conjb, \
       dim_t               m, \
       dim_t               n, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a, inc_t rs_a, inc_t cs_a, \
       ctype*     restrict b, inc_t rs_b, inc_t cs_b, \
       ctype*     restrict beta, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	/* NOTE: This microkernel can actually handle arbitrarily large
       values of m, n, and k. */ \
\
	if ( bli_is_noconj( conja ) && bli_is_noconj( conjb ) ) \
	{ \
		/* Traverse c by rows. */ \
		for ( dim_t i = 0; i < m; ++i ) \
		{ \
			ctype* restrict ci = &c[ i*rs_c ]; \
			ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype* restrict cij = &ci[ j*cs_c ]; \
				ctype* restrict bj  = &b [ j*cs_b ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,dots)( *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
	else if ( bli_is_noconj( conja ) && bli_is_conj( conjb ) ) \
	{ \
		/* Traverse c by rows. */ \
		for ( dim_t i = 0; i < m; ++i ) \
		{ \
			ctype* restrict ci = &c[ i*rs_c ]; \
			ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype* restrict cij = &ci[ j*cs_c ]; \
				ctype* restrict bj  = &b [ j*cs_b ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,axpyjs)( *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
	else if ( bli_is_conj( conja ) && bli_is_noconj( conjb ) ) \
	{ \
		/* Traverse c by rows. */ \
		for ( dim_t i = 0; i < m; ++i ) \
		{ \
			ctype* restrict ci = &c[ i*rs_c ]; \
			ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype* restrict cij = &ci[ j*cs_c ]; \
				ctype* restrict bj  = &b [ j*cs_b ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,dotjs)( *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_conj( conja ) && bli_is_conj( conjb ) ) */ \
	{ \
		/* Traverse c by rows. */ \
		for ( dim_t i = 0; i < m; ++i ) \
		{ \
			ctype* restrict ci = &c[ i*rs_c ]; \
			ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype* restrict cij = &ci[ j*cs_c ]; \
				ctype* restrict bj  = &b [ j*cs_b ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,dots)( *aij, *bij, ab ); \
				} \
\
				/* Conjugate the result to simulate conj(a^T) * conj(b). */ \
				PASTEMAC(ch,conjs)( ab ); \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC2( gemmsup_r, _armsve, _ref2 )

//
// -- Column storage case ------------------------------------------------------
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       conj_t              conja, \
       conj_t              conjb, \
       dim_t               m, \
       dim_t               n, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a, inc_t rs_a, inc_t cs_a, \
       ctype*     restrict b, inc_t rs_b, inc_t cs_b, \
       ctype*     restrict beta, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	/* NOTE: This microkernel can actually handle arbitrarily large
       values of m, n, and k. */ \
\
	if ( bli_is_noconj( conja ) && bli_is_noconj( conjb ) ) \
	{ \
		/* Traverse c by columns. */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict cj = &c[ j*cs_c ]; \
			ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				ctype* restrict cij = &cj[ i*rs_c ]; \
				ctype* restrict ai  = &a [ i*rs_a ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,dots)( *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
	else if ( bli_is_noconj( conja ) && bli_is_conj( conjb ) ) \
	{ \
		/* Traverse c by columns. */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict cj = &c[ j*cs_c ]; \
			ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				ctype* restrict cij = &cj[ i*rs_c ]; \
				ctype* restrict ai  = &a [ i*rs_a ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,axpyjs)( *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
	else if ( bli_is_conj( conja ) && bli_is_noconj( conjb ) ) \
	{ \
		/* Traverse c by columns. */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict cj = &c[ j*cs_c ]; \
			ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				ctype* restrict cij = &cj[ i*rs_c ]; \
				ctype* restrict ai  = &a [ i*rs_a ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,dotjs)( *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_conj( conja ) && bli_is_conj( conjb ) ) */ \
	{ \
		/* Traverse c by columns. */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict cj = &c[ j*cs_c ]; \
			ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				ctype* restrict cij = &cj[ i*rs_c ]; \
				ctype* restrict ai  = &a [ i*rs_a ]; \
				ctype           ab; \
\
				PASTEMAC(ch,set0s)( ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					ctype* restrict aij = &ai[ l*cs_a ]; \
					ctype* restrict bij = &bj[ l*rs_b ]; \
\
					PASTEMAC(ch,dots)( *aij, *bij, ab ); \
				} \
\
				/* Conjugate the result to simulate conj(a^T) * conj(b). */ \
				PASTEMAC(ch,conjs)( ab ); \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( PASTEMAC(ch,eq1)( *beta ) ) \
				{ \
					PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
				} \
				else if ( PASTEMAC(ch,eq0)( *beta ) ) \
				{ \
					PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
				} \
				else \
				{ \
					PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC2( gemmsup_c, _armsve, _ref2 )

