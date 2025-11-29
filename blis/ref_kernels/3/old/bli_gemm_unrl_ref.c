/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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
// -- dgemm --------------------------------------------------------------------
//

#undef     CH
#define    CH d
#undef  CTYPE
#define CTYPE double
#undef   ZERO
#define  ZERO 0.0
#undef     MR
#define    MR 4
#undef     NR
#define    NR 8

//void PASTEMAC4(CH,gemm,BLIS_CNAME_INFIX,BLIS_REF_SUF,_4x8)
void PASTEMAC6(CH,gemm,BLIS_CNAME_REF_SUFFIX,_,MR,x,NR)
     (
       dim_t               k,
       CTYPE*     restrict alpha,
       CTYPE*     restrict a,
       CTYPE*     restrict b,
       CTYPE*     restrict beta,
       CTYPE*     restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	const dim_t cs_a = MR;
	const dim_t rs_b = NR;

	CTYPE  ab00 = ZERO, ab01 = ZERO, ab02 = ZERO, ab03 = ZERO;
	CTYPE  ab10 = ZERO, ab11 = ZERO, ab12 = ZERO, ab13 = ZERO;
	CTYPE  ab20 = ZERO, ab21 = ZERO, ab22 = ZERO, ab23 = ZERO;
	CTYPE  ab30 = ZERO, ab31 = ZERO, ab32 = ZERO, ab33 = ZERO;

	CTYPE  ab04 = ZERO, ab05 = ZERO, ab06 = ZERO, ab07 = ZERO;
	CTYPE  ab14 = ZERO, ab15 = ZERO, ab16 = ZERO, ab17 = ZERO;
	CTYPE  ab24 = ZERO, ab25 = ZERO, ab26 = ZERO, ab27 = ZERO;
	CTYPE  ab34 = ZERO, ab35 = ZERO, ab36 = ZERO, ab37 = ZERO;

	// Perform a series of k rank-1 updates into ab.
	for ( ; k != 0; --k )
	{
		const CTYPE a0 = a[0];

		ab00 += a0*b[0]; ab01 += a0*b[1]; ab02 += a0*b[2]; ab03 += a0*b[3];
		ab04 += a0*b[4]; ab05 += a0*b[5]; ab06 += a0*b[6]; ab07 += a0*b[7];

		const CTYPE a1 = a[1];

		ab10 += a1*b[0]; ab11 += a1*b[1]; ab12 += a1*b[2]; ab13 += a1*b[3];
		ab14 += a1*b[4]; ab15 += a1*b[5]; ab16 += a1*b[6]; ab17 += a1*b[7];

		const CTYPE a2 = a[2];

		ab20 += a2*b[0]; ab21 += a2*b[1]; ab22 += a2*b[2]; ab23 += a2*b[3];
		ab24 += a2*b[4]; ab25 += a2*b[5]; ab26 += a2*b[6]; ab27 += a2*b[7];

		const CTYPE a3 = a[3];

		ab30 += a3*b[0]; ab31 += a3*b[1]; ab32 += a3*b[2]; ab33 += a3*b[3];
		ab34 += a3*b[4]; ab35 += a3*b[5]; ab36 += a3*b[6]; ab37 += a3*b[7];

		a += cs_a;
		b += rs_b;
	}

	// Scale each element of ab by alpha.
	if ( !PASTEMAC(CH,eq1)( *alpha ) )
	{
		const CTYPE alpha0 = *alpha;

		PASTEMAC(CH,scals)( alpha0, ab00 );
		PASTEMAC(CH,scals)( alpha0, ab01 );
		PASTEMAC(CH,scals)( alpha0, ab02 );
		PASTEMAC(CH,scals)( alpha0, ab02 );

		PASTEMAC(CH,scals)( alpha0, ab04 );
		PASTEMAC(CH,scals)( alpha0, ab05 );
		PASTEMAC(CH,scals)( alpha0, ab06 );
		PASTEMAC(CH,scals)( alpha0, ab07 );

		PASTEMAC(CH,scals)( alpha0, ab10 );
		PASTEMAC(CH,scals)( alpha0, ab11 );
		PASTEMAC(CH,scals)( alpha0, ab12 );
		PASTEMAC(CH,scals)( alpha0, ab12 );

		PASTEMAC(CH,scals)( alpha0, ab14 );
		PASTEMAC(CH,scals)( alpha0, ab15 );
		PASTEMAC(CH,scals)( alpha0, ab16 );
		PASTEMAC(CH,scals)( alpha0, ab17 );

		PASTEMAC(CH,scals)( alpha0, ab20 );
		PASTEMAC(CH,scals)( alpha0, ab21 );
		PASTEMAC(CH,scals)( alpha0, ab22 );
		PASTEMAC(CH,scals)( alpha0, ab22 );

		PASTEMAC(CH,scals)( alpha0, ab24 );
		PASTEMAC(CH,scals)( alpha0, ab25 );
		PASTEMAC(CH,scals)( alpha0, ab26 );
		PASTEMAC(CH,scals)( alpha0, ab27 );

		PASTEMAC(CH,scals)( alpha0, ab30 );
		PASTEMAC(CH,scals)( alpha0, ab31 );
		PASTEMAC(CH,scals)( alpha0, ab32 );
		PASTEMAC(CH,scals)( alpha0, ab32 );

		PASTEMAC(CH,scals)( alpha0, ab34 );
		PASTEMAC(CH,scals)( alpha0, ab35 );
		PASTEMAC(CH,scals)( alpha0, ab36 );
		PASTEMAC(CH,scals)( alpha0, ab37 );
	}

	// Output/accumulate intermediate result ab based on the storage
	// of c and the value of beta.
	if ( cs_c == 1 )
	{
		// C is row-stored.

		if ( PASTEMAC(CH,eq0)( *beta ) )
		{
			// beta == 0:
			//   c := ab

			PASTEMAC(CH,copys)( ab00, c[ 0*rs_c + 0 ] );
			PASTEMAC(CH,copys)( ab01, c[ 0*rs_c + 1 ] );
			PASTEMAC(CH,copys)( ab02, c[ 0*rs_c + 2 ] );
			PASTEMAC(CH,copys)( ab03, c[ 0*rs_c + 3 ] );

			PASTEMAC(CH,copys)( ab04, c[ 0*rs_c + 4 ] );
			PASTEMAC(CH,copys)( ab05, c[ 0*rs_c + 5 ] );
			PASTEMAC(CH,copys)( ab06, c[ 0*rs_c + 6 ] );
			PASTEMAC(CH,copys)( ab07, c[ 0*rs_c + 7 ] );

			PASTEMAC(CH,copys)( ab10, c[ 1*rs_c + 0 ] );
			PASTEMAC(CH,copys)( ab11, c[ 1*rs_c + 1 ] );
			PASTEMAC(CH,copys)( ab12, c[ 1*rs_c + 2 ] );
			PASTEMAC(CH,copys)( ab13, c[ 1*rs_c + 3 ] );

			PASTEMAC(CH,copys)( ab14, c[ 1*rs_c + 4 ] );
			PASTEMAC(CH,copys)( ab15, c[ 1*rs_c + 5 ] );
			PASTEMAC(CH,copys)( ab16, c[ 1*rs_c + 6 ] );
			PASTEMAC(CH,copys)( ab17, c[ 1*rs_c + 7 ] );

			PASTEMAC(CH,copys)( ab20, c[ 2*rs_c + 0 ] );
			PASTEMAC(CH,copys)( ab21, c[ 2*rs_c + 1 ] );
			PASTEMAC(CH,copys)( ab22, c[ 2*rs_c + 2 ] );
			PASTEMAC(CH,copys)( ab23, c[ 2*rs_c + 3 ] );

			PASTEMAC(CH,copys)( ab24, c[ 2*rs_c + 4 ] );
			PASTEMAC(CH,copys)( ab25, c[ 2*rs_c + 5 ] );
			PASTEMAC(CH,copys)( ab26, c[ 2*rs_c + 6 ] );
			PASTEMAC(CH,copys)( ab27, c[ 2*rs_c + 7 ] );

			PASTEMAC(CH,copys)( ab30, c[ 3*rs_c + 0 ] );
			PASTEMAC(CH,copys)( ab31, c[ 3*rs_c + 1 ] );
			PASTEMAC(CH,copys)( ab32, c[ 3*rs_c + 2 ] );
			PASTEMAC(CH,copys)( ab33, c[ 3*rs_c + 3 ] );

			PASTEMAC(CH,copys)( ab34, c[ 3*rs_c + 4 ] );
			PASTEMAC(CH,copys)( ab35, c[ 3*rs_c + 5 ] );
			PASTEMAC(CH,copys)( ab36, c[ 3*rs_c + 6 ] );
			PASTEMAC(CH,copys)( ab37, c[ 3*rs_c + 7 ] );
		}
		else
		{
			const CTYPE beta0 = *beta;

			// beta != 0:
			//   c := beta * c + ab

			PASTEMAC(CH,xpbys)( ab00, beta0, c[ 0*rs_c + 0 ] );
			PASTEMAC(CH,xpbys)( ab01, beta0, c[ 0*rs_c + 1 ] );
			PASTEMAC(CH,xpbys)( ab02, beta0, c[ 0*rs_c + 2 ] );
			PASTEMAC(CH,xpbys)( ab03, beta0, c[ 0*rs_c + 3 ] );

			PASTEMAC(CH,xpbys)( ab04, beta0, c[ 0*rs_c + 4 ] );
			PASTEMAC(CH,xpbys)( ab05, beta0, c[ 0*rs_c + 5 ] );
			PASTEMAC(CH,xpbys)( ab06, beta0, c[ 0*rs_c + 6 ] );
			PASTEMAC(CH,xpbys)( ab07, beta0, c[ 0*rs_c + 7 ] );

			PASTEMAC(CH,xpbys)( ab10, beta0, c[ 1*rs_c + 0 ] );
			PASTEMAC(CH,xpbys)( ab11, beta0, c[ 1*rs_c + 1 ] );
			PASTEMAC(CH,xpbys)( ab12, beta0, c[ 1*rs_c + 2 ] );
			PASTEMAC(CH,xpbys)( ab13, beta0, c[ 1*rs_c + 3 ] );

			PASTEMAC(CH,xpbys)( ab14, beta0, c[ 1*rs_c + 4 ] );
			PASTEMAC(CH,xpbys)( ab15, beta0, c[ 1*rs_c + 5 ] );
			PASTEMAC(CH,xpbys)( ab16, beta0, c[ 1*rs_c + 6 ] );
			PASTEMAC(CH,xpbys)( ab17, beta0, c[ 1*rs_c + 7 ] );

			PASTEMAC(CH,xpbys)( ab20, beta0, c[ 2*rs_c + 0 ] );
			PASTEMAC(CH,xpbys)( ab21, beta0, c[ 2*rs_c + 1 ] );
			PASTEMAC(CH,xpbys)( ab22, beta0, c[ 2*rs_c + 2 ] );
			PASTEMAC(CH,xpbys)( ab23, beta0, c[ 2*rs_c + 3 ] );

			PASTEMAC(CH,xpbys)( ab24, beta0, c[ 2*rs_c + 4 ] );
			PASTEMAC(CH,xpbys)( ab25, beta0, c[ 2*rs_c + 5 ] );
			PASTEMAC(CH,xpbys)( ab26, beta0, c[ 2*rs_c + 6 ] );
			PASTEMAC(CH,xpbys)( ab27, beta0, c[ 2*rs_c + 7 ] );

			PASTEMAC(CH,xpbys)( ab30, beta0, c[ 3*rs_c + 0 ] );
			PASTEMAC(CH,xpbys)( ab31, beta0, c[ 3*rs_c + 1 ] );
			PASTEMAC(CH,xpbys)( ab32, beta0, c[ 3*rs_c + 2 ] );
			PASTEMAC(CH,xpbys)( ab33, beta0, c[ 3*rs_c + 3 ] );

			PASTEMAC(CH,xpbys)( ab34, beta0, c[ 3*rs_c + 4 ] );
			PASTEMAC(CH,xpbys)( ab35, beta0, c[ 3*rs_c + 5 ] );
			PASTEMAC(CH,xpbys)( ab36, beta0, c[ 3*rs_c + 6 ] );
			PASTEMAC(CH,xpbys)( ab37, beta0, c[ 3*rs_c + 7 ] );
		}
	}
	else
	{
		// C is general-stored (or column-stored).

		if ( PASTEMAC(CH,eq0)( *beta ) )
		{
			// beta == 0:
			//   c := ab

			PASTEMAC(CH,copys)( ab00, c[ 0*rs_c + 0*cs_c ] );
			PASTEMAC(CH,copys)( ab01, c[ 0*rs_c + 1*cs_c ] );
			PASTEMAC(CH,copys)( ab02, c[ 0*rs_c + 2*cs_c ] );
			PASTEMAC(CH,copys)( ab03, c[ 0*rs_c + 3*cs_c ] );

			PASTEMAC(CH,copys)( ab04, c[ 0*rs_c + 4*cs_c ] );
			PASTEMAC(CH,copys)( ab05, c[ 0*rs_c + 5*cs_c ] );
			PASTEMAC(CH,copys)( ab06, c[ 0*rs_c + 6*cs_c ] );
			PASTEMAC(CH,copys)( ab07, c[ 0*rs_c + 7*cs_c ] );

			PASTEMAC(CH,copys)( ab10, c[ 1*rs_c + 0*cs_c ] );
			PASTEMAC(CH,copys)( ab11, c[ 1*rs_c + 1*cs_c ] );
			PASTEMAC(CH,copys)( ab12, c[ 1*rs_c + 2*cs_c ] );
			PASTEMAC(CH,copys)( ab13, c[ 1*rs_c + 3*cs_c ] );

			PASTEMAC(CH,copys)( ab14, c[ 1*rs_c + 4*cs_c ] );
			PASTEMAC(CH,copys)( ab15, c[ 1*rs_c + 5*cs_c ] );
			PASTEMAC(CH,copys)( ab16, c[ 1*rs_c + 6*cs_c ] );
			PASTEMAC(CH,copys)( ab17, c[ 1*rs_c + 7*cs_c ] );

			PASTEMAC(CH,copys)( ab20, c[ 2*rs_c + 0*cs_c ] );
			PASTEMAC(CH,copys)( ab21, c[ 2*rs_c + 1*cs_c ] );
			PASTEMAC(CH,copys)( ab22, c[ 2*rs_c + 2*cs_c ] );
			PASTEMAC(CH,copys)( ab23, c[ 2*rs_c + 3*cs_c ] );

			PASTEMAC(CH,copys)( ab24, c[ 2*rs_c + 4*cs_c ] );
			PASTEMAC(CH,copys)( ab25, c[ 2*rs_c + 5*cs_c ] );
			PASTEMAC(CH,copys)( ab26, c[ 2*rs_c + 6*cs_c ] );
			PASTEMAC(CH,copys)( ab27, c[ 2*rs_c + 7*cs_c ] );

			PASTEMAC(CH,copys)( ab30, c[ 3*rs_c + 0*cs_c ] );
			PASTEMAC(CH,copys)( ab31, c[ 3*rs_c + 1*cs_c ] );
			PASTEMAC(CH,copys)( ab32, c[ 3*rs_c + 2*cs_c ] );
			PASTEMAC(CH,copys)( ab33, c[ 3*rs_c + 3*cs_c ] );

			PASTEMAC(CH,copys)( ab34, c[ 3*rs_c + 4*cs_c ] );
			PASTEMAC(CH,copys)( ab35, c[ 3*rs_c + 5*cs_c ] );
			PASTEMAC(CH,copys)( ab36, c[ 3*rs_c + 6*cs_c ] );
			PASTEMAC(CH,copys)( ab37, c[ 3*rs_c + 7*cs_c ] );
		}
		else
		{
			const CTYPE beta0 = *beta;

			// beta != 0:
			//   c := beta * c + ab

			PASTEMAC(CH,xpbys)( ab00, beta0, c[ 0*rs_c + 0*cs_c ] );
			PASTEMAC(CH,xpbys)( ab01, beta0, c[ 0*rs_c + 1*cs_c ] );
			PASTEMAC(CH,xpbys)( ab02, beta0, c[ 0*rs_c + 2*cs_c ] );
			PASTEMAC(CH,xpbys)( ab03, beta0, c[ 0*rs_c + 3*cs_c ] );

			PASTEMAC(CH,xpbys)( ab04, beta0, c[ 0*rs_c + 4*cs_c ] );
			PASTEMAC(CH,xpbys)( ab05, beta0, c[ 0*rs_c + 5*cs_c ] );
			PASTEMAC(CH,xpbys)( ab06, beta0, c[ 0*rs_c + 6*cs_c ] );
			PASTEMAC(CH,xpbys)( ab07, beta0, c[ 0*rs_c + 7*cs_c ] );

			PASTEMAC(CH,xpbys)( ab10, beta0, c[ 1*rs_c + 0*cs_c ] );
			PASTEMAC(CH,xpbys)( ab11, beta0, c[ 1*rs_c + 1*cs_c ] );
			PASTEMAC(CH,xpbys)( ab12, beta0, c[ 1*rs_c + 2*cs_c ] );
			PASTEMAC(CH,xpbys)( ab13, beta0, c[ 1*rs_c + 3*cs_c ] );

			PASTEMAC(CH,xpbys)( ab14, beta0, c[ 1*rs_c + 4*cs_c ] );
			PASTEMAC(CH,xpbys)( ab15, beta0, c[ 1*rs_c + 5*cs_c ] );
			PASTEMAC(CH,xpbys)( ab16, beta0, c[ 1*rs_c + 6*cs_c ] );
			PASTEMAC(CH,xpbys)( ab17, beta0, c[ 1*rs_c + 7*cs_c ] );

			PASTEMAC(CH,xpbys)( ab20, beta0, c[ 2*rs_c + 0*cs_c ] );
			PASTEMAC(CH,xpbys)( ab21, beta0, c[ 2*rs_c + 1*cs_c ] );
			PASTEMAC(CH,xpbys)( ab22, beta0, c[ 2*rs_c + 2*cs_c ] );
			PASTEMAC(CH,xpbys)( ab23, beta0, c[ 2*rs_c + 3*cs_c ] );

			PASTEMAC(CH,xpbys)( ab24, beta0, c[ 2*rs_c + 4*cs_c ] );
			PASTEMAC(CH,xpbys)( ab25, beta0, c[ 2*rs_c + 5*cs_c ] );
			PASTEMAC(CH,xpbys)( ab26, beta0, c[ 2*rs_c + 6*cs_c ] );
			PASTEMAC(CH,xpbys)( ab27, beta0, c[ 2*rs_c + 7*cs_c ] );

			PASTEMAC(CH,xpbys)( ab30, beta0, c[ 3*rs_c + 0*cs_c ] );
			PASTEMAC(CH,xpbys)( ab31, beta0, c[ 3*rs_c + 1*cs_c ] );
			PASTEMAC(CH,xpbys)( ab32, beta0, c[ 3*rs_c + 2*cs_c ] );
			PASTEMAC(CH,xpbys)( ab33, beta0, c[ 3*rs_c + 3*cs_c ] );

			PASTEMAC(CH,xpbys)( ab34, beta0, c[ 3*rs_c + 4*cs_c ] );
			PASTEMAC(CH,xpbys)( ab35, beta0, c[ 3*rs_c + 5*cs_c ] );
			PASTEMAC(CH,xpbys)( ab36, beta0, c[ 3*rs_c + 6*cs_c ] );
			PASTEMAC(CH,xpbys)( ab37, beta0, c[ 3*rs_c + 7*cs_c ] );
		}
	}
}

