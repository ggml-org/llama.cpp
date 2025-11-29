/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2024, Southern Methodist University

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


#undef  GENTFUNCRO
#define GENTFUNCRO( ctype_r, ctype, chr, ch, opname, arch, suf, diagop ) \
\
static void PASTEMAC(chr,opname,arch,suf) \
     ( \
             dim_t           m, \
             dim_t           n, \
       const ctype* restrict a, inc_t rs_a, inc_t cs_a, \
       const ctype* restrict b, inc_t rs_b, inc_t cs_b, \
             ctype* restrict c, inc_t rs_c, inc_t cs_c, \
             pack_t          schema_b \
     ) \
{ \
	const inc_t ld_a  = cs_a; \
	const inc_t ld_b  = rs_b; \
\
	if ( bli_is_1e_packed( schema_b ) ) \
	{ \
		const inc_t       rs_a2 = 1 * rs_a; \
		const inc_t       cs_a2 = 2 * cs_a; \
		const inc_t       rs_b2 = 2 * rs_b; \
		const inc_t       cs_b2 = 2 * cs_b; \
\
		ctype_r* restrict a_r   = ( ctype_r* )a; \
		ctype_r* restrict a_i   = ( ctype_r* )a + ld_a; \
\
		ctype_r* restrict b_ri  = ( ctype_r* )b; \
		ctype_r* restrict b_ir  = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = iter; \
			dim_t n_behind  = i; \
\
			ctype_r* restrict alpha11_r = a_r  + (i  )*rs_a2 + (i  )*cs_a2; \
			ctype_r* restrict alpha11_i = a_i  + (i  )*rs_a2 + (i  )*cs_a2; \
			ctype_r* restrict a10t_r    = a_r  + (i  )*rs_a2 + (0  )*cs_a2; \
			ctype_r* restrict a10t_i    = a_i  + (i  )*rs_a2 + (0  )*cs_a2; \
			ctype_r* restrict b1_ri     = b_ri + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict b1_ir     = b_ir + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict B0_ri     = b_ri + (0  )*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a10t * B0; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_ri_r = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ri_i = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict beta11_ir_r = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ir_i = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict b01_ri      = B0_ri + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11     = c     + (i  )*rs_c  + (j  )*cs_c; \
				ctype_r           beta11c_r   = *beta11_ri_r; \
				ctype_r           beta11c_i   = *beta11_ri_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				bli_tset0ris( ch, rho11_r, \
				                      rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					ctype_r* restrict alpha10_r = a10t_r  + (l  )*cs_a2; \
					ctype_r* restrict alpha10_i = a10t_i  + (l  )*cs_a2; \
					ctype_r* restrict beta01_r  = b01_ri  + (l  )*rs_b2 + 0*cs_b; \
					ctype_r* restrict beta01_i  = b01_ri  + (l  )*rs_b2 + 1*cs_b; \
\
					bli_taxpyris( ch,ch,ch,ch, *alpha10_r, \
					                           *alpha10_i, \
					                           *beta01_r, \
					                           *beta01_i, \
					                           rho11_r, \
					                           rho11_i ); \
				} \
				bli_tsubris( ch,ch,ch, rho11_r, \
				                       rho11_i, \
				                       beta11c_r, \
				                       beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(t,diagop)( ch,ch,ch, *alpha11_r, \
				                              *alpha11_i, \
				                              beta11c_r, \
				                              beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				bli_tsets( ch,ch,  beta11c_r, beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
				{ \
					bli_tcopyris( ch,ch,  beta11c_r, beta11c_i, *(beta11_ri_r + d), *(beta11_ri_i + d) ); \
					bli_tcopyris( ch,ch, -beta11c_i, beta11c_r, *(beta11_ir_r + d), *(beta11_ir_i + d) ); \
                } \
			} \
		} \
	} \
	else /* ( bli_is_1r_packed( schema_b ) ) */ \
	{ \
		const inc_t       rs_b2 = 2 * rs_b; \
		const inc_t       cs_b2 = 1 * cs_b; \
\
		ctype*   restrict a_ri  = ( ctype*   )a; \
		/*ctype*   restrict a_ir  = ( ctype*   )a + ld_a/2;*/ \
\
		ctype_r* restrict b_r   = ( ctype_r* )b; \
		ctype_r* restrict b_i   = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = iter; \
			dim_t n_behind  = i; \
\
			ctype*   restrict alpha11_ri = a_ri + (i  )*rs_a  + (i  )*cs_a; \
			ctype_r* restrict alpha11_r  = &PASTEMAC(ch,real)( *alpha11_ri ); \
			ctype_r* restrict alpha11_i  = &PASTEMAC(ch,imag)( *alpha11_ri ); \
			ctype*   restrict a10t_ri    = a_ri + (i  )*rs_a  + (0  )*cs_a; \
			ctype_r* restrict b1_r       = b_r  + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict b1_i       = b_i  + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict B0_r       = b_r  + (0  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict B0_i       = b_i  + (0  )*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a10t * B0; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_r  = b1_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict beta11_i  = b1_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b01_r     = B0_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b01_i     = B0_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11   = c    + (i  )*rs_c  + (j  )*cs_c; \
				ctype_r           beta11c_r = *beta11_r; \
				ctype_r           beta11c_i = *beta11_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				bli_tset0ris( ch, rho11_r, \
				                  rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					ctype*   restrict alpha10_ri = a10t_ri + (l  )*cs_a; \
					ctype_r* restrict alpha10_r  = &PASTEMAC(ch,real)( *alpha10_ri ); \
					ctype_r* restrict alpha10_i  = &PASTEMAC(ch,imag)( *alpha10_ri ); \
					ctype_r* restrict beta01_r   = b01_r   + (l  )*rs_b2; \
					ctype_r* restrict beta01_i   = b01_i   + (l  )*rs_b2; \
\
					bli_taxpyris( ch,ch,ch,ch, *alpha10_r, \
					                           *alpha10_i, \
					                           *beta01_r, \
					                           *beta01_i, \
					                           rho11_r, \
					                           rho11_i ); \
				} \
				bli_tsubris( ch,ch,ch, rho11_r, \
				                       rho11_i, \
				                       beta11c_r, \
				                       beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(t,diagop)( ch,ch,ch, *alpha11_r, \
				                              *alpha11_i, \
				                              beta11c_r, \
				                              beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				bli_tsets( ch,ch, beta11c_r, \
				                  beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
					bli_tcopyris( ch,ch, beta11c_r, \
					                     beta11c_i, \
					                     *(beta11_r + d), \
					                     *(beta11_i + d) ); \
			} \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNCRO( trsm1m_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scalris )
#else
INSERT_GENTFUNCRO( trsm1m_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscalris )
#endif


#undef  GENTFUNCRO
#define GENTFUNCRO( ctype_r, ctype, chr, ch, opname, arch, suf, diagop ) \
\
static void PASTEMAC(chr,opname,arch,suf) \
     ( \
             dim_t           m, \
             dim_t           n, \
       const ctype* restrict a, inc_t rs_a, inc_t cs_a, \
       const ctype* restrict b, inc_t rs_b, inc_t cs_b, \
             ctype* restrict c, inc_t rs_c, inc_t cs_c, \
             pack_t          schema_b \
     ) \
{ \
	const inc_t ld_a  = cs_a; \
	const inc_t ld_b  = rs_b; \
\
	if ( bli_is_1e_packed( schema_b ) ) \
	{ \
		const inc_t       rs_a2 = 1 * rs_a; \
		const inc_t       cs_a2 = 2 * cs_a; \
		const inc_t       rs_b2 = 2 * rs_b; \
		const inc_t       cs_b2 = 2 * cs_b; \
\
		ctype_r* restrict a_r   = ( ctype_r* )a; \
		ctype_r* restrict a_i   = ( ctype_r* )a + ld_a; \
\
		ctype_r* restrict b_ri  = ( ctype_r* )b; \
		ctype_r* restrict b_ir  = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = m - iter - 1; \
			dim_t n_behind  = iter; \
\
			ctype_r* restrict alpha11_r  = a_r  + (i  )*rs_a2 + (i  )*cs_a2; \
			ctype_r* restrict alpha11_i  = a_i  + (i  )*rs_a2 + (i  )*cs_a2; \
			ctype_r* restrict a12t_r     = a_r  + (i  )*rs_a2 + (i+1)*cs_a2; \
			ctype_r* restrict a12t_i     = a_i  + (i  )*rs_a2 + (i+1)*cs_a2; \
			ctype_r* restrict b1_ri      = b_ri + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict b1_ir      = b_ir + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict B2_ri      = b_ri + (i+1)*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a12t * B2; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_ri_r = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ri_i = b1_ri + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict beta11_ir_r = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 0*cs_b; \
				ctype_r* restrict beta11_ir_i = b1_ir + (0  )*rs_b2 + (j  )*cs_b2 + 1*cs_b; \
				ctype_r* restrict b21_ri      = B2_ri + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11     = c     + (i  )*rs_c + (j  )*cs_c; \
				ctype_r           beta11c_r   = *beta11_ri_r; \
				ctype_r           beta11c_i   = *beta11_ri_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				bli_tset0ris( ch, rho11_r, \
				                      rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					ctype_r* restrict alpha12_r = a12t_r + (l  )*cs_a2; \
					ctype_r* restrict alpha12_i = a12t_i + (l  )*cs_a2; \
					ctype_r* restrict beta21_r  = b21_ri + (l  )*rs_b2 + 0*cs_b; \
					ctype_r* restrict beta21_i  = b21_ri + (l  )*rs_b2 + 1*cs_b; \
\
					bli_taxpyris( ch,ch,ch,ch, *alpha12_r, \
					                           *alpha12_i, \
					                           *beta21_r, \
					                           *beta21_i, \
					                           rho11_r, \
					                           rho11_i ); \
				} \
				bli_tsubris( ch,ch,ch, rho11_r, \
				                       rho11_i, \
				                       beta11c_r, \
				                       beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(t,diagop)( ch,ch,ch, *alpha11_r, \
				                              *alpha11_i, \
				                              beta11c_r, \
				                              beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				bli_tsets( ch,ch, beta11c_r, beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
				{ \
					bli_tcopyris( ch,ch,  beta11c_r, beta11c_i, *(beta11_ri_r + d), *(beta11_ri_i + d) ); \
					bli_tcopyris( ch,ch, -beta11c_i, beta11c_r, *(beta11_ir_r + d), *(beta11_ir_i + d) ); \
                } \
			} \
		} \
	} \
	else /* if ( bli_is_1r_packed( schema_b ) ) */ \
	{ \
		const inc_t       rs_b2 = 2 * rs_b; \
		const inc_t       cs_b2 = 1 * cs_b; \
\
		ctype*   restrict a_ri  = ( ctype*   )a; \
		/*ctype*   restrict a_ir  = ( ctype*   )a + ld_a/2;*/ \
\
		ctype_r* restrict b_r   = ( ctype_r* )b; \
		ctype_r* restrict b_i   = ( ctype_r* )b + ld_b; \
\
		for ( dim_t iter = 0; iter < m; ++iter ) \
		{ \
			dim_t i         = m - iter - 1; \
			dim_t n_behind  = iter; \
\
			ctype*   restrict alpha11_ri = a_ri + (i  )*rs_a  + (i  )*cs_a; \
			ctype_r* restrict alpha11_r  = &PASTEMAC(ch,real)( *alpha11_ri ); \
			ctype_r* restrict alpha11_i  = &PASTEMAC(ch,imag)( *alpha11_ri ); \
			ctype*   restrict a12t_ri    = a_ri + (i  )*rs_a  + (i+1)*cs_a; \
			ctype_r* restrict b1_r       = b_r  + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict b1_i       = b_i  + (i  )*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict B2_r       = b_r  + (i+1)*rs_b2 + (0  )*cs_b2; \
			ctype_r* restrict B2_i       = b_i  + (i+1)*rs_b2 + (0  )*cs_b2; \
\
			/* b1 = b1 - a12t * B2; */ \
			/* b1 = b1 / alpha11; */ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				ctype_r* restrict beta11_r  = b1_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict beta11_i  = b1_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b21_r     = B2_r + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype_r* restrict b21_i     = B2_i + (0  )*rs_b2 + (j  )*cs_b2; \
				ctype*   restrict gamma11   = c    + (i  )*rs_c  + (j  )*cs_c; \
				ctype_r           beta11c_r = *beta11_r; \
				ctype_r           beta11c_i = *beta11_i; \
				ctype_r           rho11_r; \
				ctype_r           rho11_i; \
\
				/* beta11 = beta11 - a10t * b01; */ \
				bli_tset0ris( ch, rho11_r, \
				                      rho11_i ); \
				for ( dim_t l = 0; l < n_behind; ++l ) \
				{ \
					ctype*   restrict alpha12_ri = a12t_ri + (l  )*cs_a; \
					ctype_r* restrict alpha12_r  = &PASTEMAC(ch,real)( *alpha12_ri ); \
					ctype_r* restrict alpha12_i  = &PASTEMAC(ch,imag)( *alpha12_ri ); \
					ctype_r* restrict beta21_r   = b21_r   + (l  )*rs_b2; \
					ctype_r* restrict beta21_i   = b21_i   + (l  )*rs_b2; \
\
					bli_taxpyris( ch,ch,ch,ch, *alpha12_r, \
					                           *alpha12_i, \
					                           *beta21_r, \
					                           *beta21_i, \
					                           rho11_r, \
					                           rho11_i ); \
				} \
				bli_tsubris( ch,ch,ch, rho11_r, \
				                       rho11_i, \
				                       beta11c_r, \
				                       beta11c_i ); \
\
				/* beta11 = beta11 / alpha11; */ \
				/* NOTE: When preinversion is enabled, the INVERSE of alpha11
				   (1.0/alpha11) is stored during packing instead alpha11 so we
				   can multiply rather than divide. When preinversion is disabled,
				   alpha11 is stored and division happens below explicitly. */ \
				PASTEMAC(t,diagop)( ch,ch,ch, *alpha11_r, \
				                              *alpha11_i, \
				                              beta11c_r, \
				                              beta11c_i ); \
\
				/* Output final result to matrix c. */ \
				bli_tsets( ch,ch, beta11c_r, \
				                  beta11c_i, *gamma11 ); \
\
				/* Store the local values back to b11. */ \
				for ( dim_t d = 0; d < cs_b; ++d ) \
					bli_tcopyris( ch,ch, beta11c_r, \
					                     beta11c_i, \
					                     *(beta11_r + d), \
					                     *(beta11_i + d) ); \
			} \
		} \
	} \
}

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
INSERT_GENTFUNCRO( trsm1m_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, scalris )
#else
INSERT_GENTFUNCRO( trsm1m_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, invscalris )
#endif

#undef  GENTFUNCRO
#define GENTFUNCRO( ctype_r, ctype, chr, ch, opname, arch, suf, trsmname ) \
\
void PASTEMAC(chr,opname,arch,suf) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha0, \
       const void*      a1x0, \
       const void*      a110, \
       const void*      bx10, \
             void*      b110, \
             void*      c110, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* auxinfo, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype*      alpha       = alpha0; \
	const ctype*      a1x         = a1x0; \
	const ctype*      a11         = a110; \
	const ctype*      bx1         = bx10; \
	      ctype*      b11         = b110; \
	      ctype*      c11         = c110; \
\
	const cntl_t*     params     = bli_auxinfo_params( auxinfo ); \
\
	const gemm_ukr_ft rgemm_ukr  = bli_trsm_var_cntl_real_gemm_ukr( params ); \
	const bool        row_pref   = bli_trsm_var_cntl_row_pref( params ); \
	const void*       params_r   = bli_trsm_var_cntl_real_params( params ); \
\
	const dim_t       mr         = bli_trsm_var_cntl_mr( params ); \
	const dim_t       nr         = bli_trsm_var_cntl_nr( params ); \
\
	const dim_t       mr_r       = row_pref ? mr : 2 * mr; \
	const dim_t       nr_r       = row_pref ? 2 * nr : nr; \
\
	/* Convert the micro-tile dimensions from being in units of complex elements to
	   be in units of real elements. */ \
	const dim_t       k_r         = 2 * k; \
\
	ctype             bt[ BLIS_STACK_BUF_MAX_SIZE \
	                      / sizeof( ctype ) ] \
	                      __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	inc_t             rs_bt       = row_pref ? nr : 1; \
	inc_t             cs_bt       = row_pref ? 1 : mr; \
\
	inc_t             rs_bt_r     = row_pref ? nr_r : 1; \
	inc_t             cs_bt_r     = row_pref ? 1 : mr_r; \
\
	const pack_t      schema_b    = bli_auxinfo_schema_b( auxinfo ); \
\
	ctype_r* restrict a1x_r       = ( ctype_r* )a1x; \
\
	const inc_t       rs_a        = bli_trsm_var_cntl_mr_bcast( params ); \
	const inc_t       cs_a        = bli_trsm_var_cntl_mr_pack( params ); \
\
	ctype_r* restrict bx1_r       = ( ctype_r* )bx1; \
\
	const inc_t       rs_b        = bli_trsm_var_cntl_nr_pack( params ); \
	const inc_t       cs_b        = bli_trsm_var_cntl_nr_bcast( params ); \
\
	ctype_r* restrict zero_r      = PASTEMAC(chr,0); \
	ctype_r* restrict minus_one_r = PASTEMAC(chr,m1); \
\
	const ctype_r     alpha_r     = PASTEMAC(ch,real)( *alpha ); \
	const ctype_r     alpha_i     = PASTEMAC(ch,imag)( *alpha ); ( void ) alpha_i; \
\
	auxinfo_t auxinfo_r = *auxinfo; \
	bli_auxinfo_set_params( params_r, &auxinfo_r ); \
\
\
	/* Since b11 is stored in the 1e or 1r schema, we cannot update it
	   directly, and instead must compute the matrix product in a local
	   temporary microtile and then accumulate it into b11 according to
	   its schema. */ \
\
	/* lower: bt = -1.0 * a10 * b01;
	   upper: bt = -1.0 * a12 * b21; */ \
	rgemm_ukr \
	( \
	  mr_r, \
	  nr_r, \
	  k_r, \
	  minus_one_r, \
	  a1x_r, \
	  bx1_r, \
	  zero_r, \
	  bt, rs_bt_r, cs_bt_r, \
	  &auxinfo_r, \
	  cntx  \
	); \
\
	if ( bli_is_1e_packed( schema_b ) ) \
	{ \
		const inc_t       ld_b   =     rs_b; \
		const inc_t       rs_b2  = 2 * rs_b; \
		const inc_t       cs_b2  = 2 * cs_b; \
\
		ctype_r* restrict b11_ri = ( ctype_r* )b11; \
		ctype_r* restrict b11_ir = ( ctype_r* )b11 + ld_b; \
\
		/* b11 = alpha * b11 + bt; */ \
		for ( dim_t j = 0; j < nr; ++j ) \
		for ( dim_t i = 0; i < mr; ++i ) \
		for ( dim_t d = 0; d < cs_b; ++d ) \
		{ \
			ctype*   restrict beta11t     = bt     + i*rs_bt + j*cs_bt; \
			ctype_r* restrict beta11t_r   = &PASTEMAC(ch,real)( *beta11t ); \
			ctype_r* restrict beta11t_i   = &PASTEMAC(ch,imag)( *beta11t ); \
\
			ctype_r* restrict beta11_ri_r = b11_ri + i*rs_b2 + j*cs_b2 + 0*cs_b + d; \
			ctype_r* restrict beta11_ri_i = b11_ri + i*rs_b2 + j*cs_b2 + 1*cs_b + d; \
			ctype_r* restrict beta11_ir_r = b11_ir + i*rs_b2 + j*cs_b2 + 0*cs_b + d; \
			ctype_r* restrict beta11_ir_i = b11_ir + i*rs_b2 + j*cs_b2 + 1*cs_b + d; \
\
			bli_txpbyris \
			( \
			  ch,chr,ch,chr, \
			  *beta11t_r, \
			  *beta11t_i, \
			  alpha_r, \
			  alpha_i, /* alpha_i not referenced */ \
			  *beta11_ri_r, \
			  *beta11_ri_i  \
			); \
\
			bli_tcopyris( ch,ch, -*beta11_ri_i, *beta11_ri_r, \
			                      *beta11_ir_r, *beta11_ir_i ); \
		} \
	} \
	else /* if ( bli_is_1r_packed( schema_b ) ) */ \
	{ \
		const inc_t       ld_b  =     rs_b; \
		const inc_t       rs_b2 = 2 * rs_b; \
		const inc_t       cs_b2 =     cs_b; \
\
		ctype_r* restrict b11_r = ( ctype_r* )b11; \
		ctype_r* restrict b11_i = ( ctype_r* )b11 + ld_b; \
\
		/* b11 = alpha * b11 + bt; */ \
		for ( dim_t j = 0; j < nr; ++j ) \
		for ( dim_t i = 0; i < mr; ++i ) \
		for ( dim_t d = 0; d < cs_b; ++d ) \
		{ \
			ctype*   restrict beta11t   = bt    + i*rs_bt + j*cs_bt; \
			ctype_r* restrict beta11t_r = &PASTEMAC(ch,real)( *beta11t ); \
			ctype_r* restrict beta11t_i = &PASTEMAC(ch,imag)( *beta11t ); \
\
			ctype_r* restrict beta11_r  = b11_r + i*rs_b2 + j*cs_b2 + d; \
			ctype_r* restrict beta11_i  = b11_i + i*rs_b2 + j*cs_b2 + d; \
\
			bli_txpbyris \
			( \
			  ch,chr,ch,chr, \
			  *beta11t_r, \
			  *beta11t_i, \
			  alpha_r, \
			  alpha_i, /* alpha_i not referenced */ \
			  *beta11_r, \
			  *beta11_i  \
			); \
		} \
	} \
\
	/* b11 = inv(a11) * b11;
	   c11 = b11; */ \
	PASTEMAC(chr,trsmname,arch,suf) \
	( \
	  m, n, \
	  a11, rs_a, cs_a, \
	  b11, rs_b, cs_b, \
	  c11, rs_c, cs_c, \
	  schema_b  \
	); \
}

INSERT_GENTFUNCRO( gemmtrsm1m_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, trsm1m_l )
INSERT_GENTFUNCRO( gemmtrsm1m_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, trsm1m_u )
