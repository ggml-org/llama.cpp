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

//
// -- Row storage case ---------------------------------------------------------
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
             conj_t     conja, \
             conj_t     conjb, \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha0, \
       const void*      a0, inc_t rs_a, inc_t cs_a, \
       const void*      b0, inc_t rs_b, inc_t cs_b, \
       const void*      beta0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype* restrict alpha = alpha0; \
	const ctype* restrict a     = a0; \
	const ctype* restrict b     = b0; \
	const ctype* restrict beta  = beta0; \
	      ctype* restrict c     = c0; \
\
	/* NOTE: This microkernel can actually handle arbitrarily large
	   values of m, n, and k. */ \
\
	if ( bli_is_noconj( conja ) && bli_is_noconj( conjb ) ) \
	{ \
		/* Traverse c by rows. */ \
		for ( dim_t i = 0; i < m; ++i ) \
		{ \
			      ctype* restrict ci = &c[ i*rs_c ]; \
			const ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				      ctype* restrict cij = &ci[ j*cs_c ]; \
				const ctype* restrict bj  = &b [ j*cs_b ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_tdots( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
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
			const ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				      ctype* restrict cij = &ci[ j*cs_c ]; \
				const ctype* restrict bj  = &b [ j*cs_b ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_taxpyjs( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
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
			const ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				      ctype* restrict cij = &ci[ j*cs_c ]; \
				const ctype* restrict bj  = &b [ j*cs_b ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_tdotjs( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
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
			const ctype* restrict ai = &a[ i*rs_a ]; \
\
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				      ctype* restrict cij = &ci[ j*cs_c ]; \
				const ctype* restrict bj  = &b [ j*cs_b ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_tdots( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* Conjugate the result to simulate conj(a^T) * conj(b). */ \
				bli_tconjs( ch, ab ); \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( gemmsup_r, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

//
// -- Column storage case ------------------------------------------------------
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
             conj_t     conja, \
             conj_t     conjb, \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha0, \
       const void*      a0, inc_t rs_a, inc_t cs_a, \
       const void*      b0, inc_t rs_b, inc_t cs_b, \
       const void*      beta0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype* restrict alpha = alpha0; \
	const ctype* restrict a     = a0; \
	const ctype* restrict b     = b0; \
	const ctype* restrict beta  = beta0; \
	      ctype* restrict c     = c0; \
\
	/* NOTE: This microkernel can actually handle arbitrarily large
	   values of m, n, and k. */ \
\
	if ( bli_is_noconj( conja ) && bli_is_noconj( conjb ) ) \
	{ \
		/* Traverse c by columns. */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			      ctype* restrict cj = &c[ j*cs_c ]; \
			const ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				      ctype* restrict cij = &cj[ i*rs_c ]; \
				const ctype* restrict ai  = &a [ i*rs_a ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_tdots( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
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
			const ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				      ctype* restrict cij = &cj[ i*rs_c ]; \
				const ctype* restrict ai  = &a [ i*rs_a ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_taxpyjs( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
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
			const ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				      ctype* restrict cij = &cj[ i*rs_c ]; \
				const ctype* restrict ai  = &a [ i*rs_a ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_tdotjs( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
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
			const ctype* restrict bj = &b[ j*cs_b ]; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				      ctype* restrict cij = &cj[ i*rs_c ]; \
				const ctype* restrict ai  = &a [ i*rs_a ]; \
				      ctype           ab; \
\
				bli_tset0s( ch, ab ); \
\
				/* Perform a dot product to update the (i,j) element of c. */ \
				for ( dim_t l = 0; l < k; ++l ) \
				{ \
					const ctype* restrict aij = &ai[ l*cs_a ]; \
					const ctype* restrict bij = &bj[ l*rs_b ]; \
\
					bli_tdots( ch,ch,ch,ch, *aij, *bij, ab ); \
				} \
\
				/* Conjugate the result to simulate conj(a^T) * conj(b). */ \
				bli_tconjs( ch, ab ); \
\
				/* If beta is one, add ab into c. If beta is zero, overwrite c
				   with the result in ab. Otherwise, scale by beta and accumulate
				   ab to c. */ \
				if ( bli_teq1s( ch, *beta ) ) \
				{ \
					bli_taxpys( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else if ( bli_teq0s( ch, *beta ) ) \
				{ \
					bli_tscal2s( ch,ch,ch,ch, *alpha, ab, *cij ); \
				} \
				else \
				{ \
					bli_taxpbys( ch,ch,ch,ch,ch, *alpha, ab, *beta, *cij ); \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( gemmsup_c, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

//
// -- General storage case -----------------------------------------------------
//

INSERT_GENTFUNC_BASIC( gemmsup_g, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

