/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022-2025, Southern Methodist University

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

// An implementation that indexes through B with the assumption that all
// elements were broadcast (duplicated) by a factor of NP/NR.

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, arch, suf, diagop ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
       const void*      a0, \
             void*      b0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype*    a      = a0; \
	      ctype*    b      = b0; \
	      ctype*    c      = c0; \
\
	const num_t     dt     = PASTEMAC(ch,type); \
\
	const dim_t     mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t     nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t     packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const inc_t     packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const dim_t     m      = mr; \
	const dim_t     n      = nr; \
\
	const inc_t     rs_a   = bli_cntx_get_blksz_def_dt( dt, BLIS_BBM, cntx ); \
	const inc_t     cs_a   = packmr; \
\
	const inc_t     rs_b   = packnr; \
	const inc_t     cs_b   = bli_cntx_get_blksz_def_dt( dt, BLIS_BBN, cntx ); \
\
	for ( dim_t iter = 0; iter < m; ++iter ) \
	{ \
		dim_t i        = iter; \
		dim_t n_behind = i; \
\
		const ctype_r* restrict alpha11r = ( ctype_r* )( a + (i  )*rs_a + (i  )*cs_a ); \
		const ctype_r* restrict alpha11i = alpha11r + rs_a; \
		const ctype*   restrict a10t     = a + (i  )*rs_a + (0  )*cs_a; \
		      ctype*   restrict B0       = b + (0  )*rs_b + (0  )*cs_b; \
		      ctype*   restrict b1       = b + (i  )*rs_b + (0  )*cs_b; \
\
		/* b1 = b1 - a10t * B0; */ \
		/* b1 = b1 / alpha11; */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype*   restrict b01      = B0 + (0  )*rs_b + (j  )*cs_b; \
			ctype_r* restrict beta11r  = ( ctype_r* )( b1 + (0  )*rs_b + (j  )*cs_b ); \
			ctype_r* restrict beta11i  = beta11r + cs_b; \
			ctype_r* restrict gamma11r = ( ctype_r* )( c  + (i  )*rs_c + (j  )*cs_c ); \
			ctype_r* restrict gamma11i = gamma11r + 1; \
			ctype_r           beta11cr = *beta11r; \
			ctype_r           beta11ci = *beta11i; \
			ctype_r           rho11r; \
			ctype_r           rho11i; \
\
			/* beta11 = beta11 - a10t * b01; */ \
			bli_tset0s( chr, rho11r ); \
			bli_tset0s( chr, rho11i ); \
			for ( dim_t l = 0; l < n_behind; ++l ) \
			{ \
				const ctype_r* restrict alpha10r = ( ctype_r* )( a10t + (l  )*cs_a ); \
				const ctype_r* restrict alpha10i = alpha10r + rs_a; \
				      ctype_r* restrict beta01r  = ( ctype_r* )( b01  + (l  )*rs_b ); \
				      ctype_r* restrict beta01i  = beta01r + cs_b; \
\
				bli_taxpyris( ch,ch,ch,ch, *alpha10r, *alpha10i, *beta01r, *beta01i, rho11r, rho11i ); \
			} \
			bli_tsubris( ch,ch,ch, rho11r, rho11i, beta11cr, beta11ci ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: When preinversion is enabled, the INVERSE of alpha11
			   (1.0/alpha11) is stored during packing instead alpha11 so we
			   can multiply rather than divide. When preinversion is disabled,
			   alpha11 is stored and division happens below explicitly. */ \
			PASTEMAC(t,diagop)( ch,ch,ch, *alpha11r, *alpha11i, beta11cr, beta11ci ); \
\
			/* Output final result to matrix c. */ \
			bli_tcopyris( ch,ch, beta11cr, beta11ci, *gamma11r, *gamma11i ); \
\
			/* Store the local value back to b11. */ \
			for ( dim_t d = 0; d < cs_b; ++d ) \
				bli_tcopyris( ch,ch, beta11cr, beta11ci, *(beta11r + d), *(beta11i + d) ); \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNCR_BASIC( trsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scalris )
#else
INSERT_GENTFUNCR_BASIC( trsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscalris )
#endif


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, arch, suf, diagop ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
       const void*      a0, \
             void*      b0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype*    a      = a0; \
	      ctype*    b      = b0; \
	      ctype*    c      = c0; \
\
	const num_t     dt     = PASTEMAC(ch,type); \
\
	const dim_t     mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t     nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t     packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const inc_t     packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const dim_t     m      = mr; \
	const dim_t     n      = nr; \
\
	const inc_t     rs_a   = bli_cntx_get_blksz_def_dt( dt, BLIS_BBM, cntx ); \
	const inc_t     cs_a   = packmr; \
\
	const inc_t     rs_b   = packnr; \
	const inc_t     cs_b   = bli_cntx_get_blksz_def_dt( dt, BLIS_BBN, cntx ); \
\
	for ( dim_t iter = 0; iter < m; ++iter ) \
	{ \
		dim_t i        = m - iter - 1; \
		dim_t n_behind = iter; \
\
		const ctype_r* restrict alpha11r = ( ctype_r* )( a + (i  )*rs_a + (i  )*cs_a ); \
		const ctype_r* restrict alpha11i = alpha11r + rs_a; \
		const ctype*   restrict a12t     = a + (i  )*rs_a + (i+1)*cs_a; \
		      ctype*   restrict b1       = b + (i  )*rs_b + (0  )*cs_b; \
		      ctype*   restrict B2       = b + (i+1)*rs_b + (0  )*cs_b; \
\
		/* b1 = b1 - a12t * B2; */ \
		/* b1 = b1 / alpha11; */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype_r* restrict beta11r  = ( ctype_r* )( b1 + (0  )*rs_b + (j  )*cs_b ); \
			ctype_r* restrict beta11i  = beta11r + cs_b; \
			ctype*   restrict b21      = B2 + (0  )*rs_b + (j  )*cs_b; \
			ctype_r* restrict gamma11r = ( ctype_r* )( c  + (i  )*rs_c + (j  )*cs_c ); \
			ctype_r* restrict gamma11i = gamma11r + 1; \
			ctype_r           beta11cr = *beta11r; \
			ctype_r           beta11ci = *beta11i; \
			ctype_r           rho11r; \
			ctype_r           rho11i; \
\
			/* beta11 = beta11 - a12t * b21; */ \
			bli_tset0s( chr, rho11r ); \
			bli_tset0s( chr, rho11i ); \
			for ( dim_t l = 0; l < n_behind; ++l ) \
			{ \
				const ctype_r* restrict alpha12r = ( ctype_r* )( a12t + (l  )*cs_a ); \
				const ctype_r* restrict alpha12i = alpha12r + rs_a; \
				      ctype_r* restrict beta21r  = ( ctype_r* )( b21  + (l  )*rs_b ); \
				      ctype_r* restrict beta21i  = beta21r + cs_b; \
\
				bli_taxpyris( ch,ch,ch,ch, *alpha12r, *alpha12i, *beta21r, *beta21i, rho11r, rho11i ); \
			} \
			bli_tsubris( ch,ch,ch, rho11r, rho11i, beta11cr, beta11ci ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: When preinversion is enabled, the INVERSE of alpha11
			   (1.0/alpha11) is stored during packing instead alpha11 so we
			   can multiply rather than divide. When preinversion is disabled,
			   alpha11 is stored and division happens below explicitly. */ \
			PASTEMAC(t,diagop)( ch,ch,ch, *alpha11r, *alpha11i, beta11cr, beta11ci ); \
\
			/* Output final result to matrix c. */ \
			bli_tcopyris( ch,ch, beta11cr, beta11ci, *gamma11r, *gamma11i ); \
\
			/* Store the local value back to b11. */ \
			for ( dim_t d = 0; d < cs_b; ++d ) \
				bli_tcopyris( ch,ch, beta11cr, beta11ci, *(beta11r + d), *(beta11i + d) ); \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNCR_BASIC( trsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scalris )
#else
INSERT_GENTFUNCR_BASIC( trsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscalris )
#endif

