/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

/*
   rrr:
	 --------        ------        --------
	 --------        ------        --------
	 --------   +=   ------ ...    --------
	 --------        ------        --------
	 --------        ------            :
	 --------        ------            :

   rcr:
	 --------        | | | |       --------
	 --------        | | | |       --------
	 --------   +=   | | | | ...   --------
	 --------        | | | |       --------
	 --------        | | | |           :
	 --------        | | | |           :

   Assumptions:
   - B is row-stored;
   - A is row- or column-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.

   NOTE: These kernels explicitly support column-oriented IO, implemented
   via an in-register transpose. And thus they also support the crr and
   ccr cases, though only crr is ever utilized (because ccr is handled by
   transposing the operation and executing rcr, which does not incur the
   cost of the in-register transpose).

   crr:
	 | | | | | | | |       ------        --------
	 | | | | | | | |       ------        --------
	 | | | | | | | |  +=   ------ ...    --------
	 | | | | | | | |       ------        --------
	 | | | | | | | |       ------            :
	 | | | | | | | |       ------            :
*/

// Prototype reference microkernels.
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )


// NOTE: Normally, for any "?x1" kernel, we would call the reference kernel.
// However, at least one other subconfiguration (zen) uses this kernel set, so
// we need to be able to call a set of "?x1" kernels that we know will actually
// exist regardless of which subconfiguration these kernels were used by. Thus,
// the compromise employed here is to inline the reference kernel so it gets
// compiled as part of the haswell kernel set, and hence can unconditionally be
// called by other kernels within that kernel set.

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mdim ) \
\
void PASTEMAC(ch,opname) \
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
       const cntx_t*    cntx \
     ) \
{ \
	const ctype* alpha = alpha0; \
	const ctype* a     = a0; \
	const ctype* b     = b0; \
	const ctype* beta  = beta0; \
	      ctype* c     = c0; \
\
	for ( dim_t i = 0; i < mdim; ++i ) \
	{ \
		      ctype* ci = &c[ i*rs_c ]; \
		const ctype* ai = &a[ i*rs_a ]; \
\
		/* for ( dim_t j = 0; j < 1; ++j ) */ \
		{ \
			      ctype* cij = ci /*[ j*cs_c ]*/ ; \
			const ctype* bj  = b  /*[ j*cs_b ]*/ ; \
			ctype        ab; \
\
			PASTEMAC(ch,set0s)( ab ); \
\
			/* Perform a dot product to update the (i,j) element of c. */ \
			for ( dim_t l = 0; l < k; ++l ) \
			{ \
				const ctype* aij = &ai[ l*cs_a ]; \
				const ctype* bij = &bj[ l*rs_b ]; \
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
			else if ( PASTEMAC(d,eq0)( *beta ) ) \
			{ \
				PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
			} \
			else \
			{ \
				PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
			} \
		} \
	} \
}

GENTFUNC( double, d, gemmsup_r_haswell_ref_6x1, 6 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_5x1, 5 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_4x1, 4 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_3x1, 3 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_2x1, 2 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_1x1, 1 )

