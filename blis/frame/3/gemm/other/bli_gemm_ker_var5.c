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

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)(
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   alpha,
                           void*   a, inc_t cs_a, dim_t pd_a, inc_t ps_a,
                           void*   b, inc_t rs_b, dim_t pd_b, inc_t ps_b,
                           void*   beta,
                           void*   c, inc_t rs_c, inc_t cs_c,
                           void_fp gemm_ukr
                         );

static FUNCPTR_T GENARRAY(ftypes,gemm_ker_var5);


void bli_gemm_ker_var5( obj_t*  a,
                        obj_t*  b,
                        obj_t*  c,
                        cntx_t* cntx,
                        gemm_t* cntl,
                        gemm_thrinfo_t* thread )
{
	num_t     dt_exec   = bli_obj_exec_dt( c );

	dim_t     m         = bli_obj_length( c );
	dim_t     n         = bli_obj_width( c );
	dim_t     k         = bli_obj_width( a );

	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     cs_a      = bli_obj_col_stride( a );
	dim_t     pd_a      = bli_obj_panel_dim( a );
	inc_t     ps_a      = bli_obj_panel_stride( a );

	void*     buf_b     = bli_obj_buffer_at_off( b );
	inc_t     rs_b      = bli_obj_row_stride( b );
	dim_t     pd_b      = bli_obj_panel_dim( b );
	inc_t     ps_b      = bli_obj_panel_stride( b );

	void*     buf_c     = bli_obj_buffer_at_off( c );
	inc_t     rs_c      = bli_obj_row_stride( c );
	inc_t     cs_c      = bli_obj_col_stride( c );

	obj_t     scalar_a;
	obj_t     scalar_b;

	void*     buf_alpha;
	void*     buf_beta;

	FUNCPTR_T f;

	func_t*   gemm_ukrs;
	void_fp   gemm_ukr;


	// Detach and multiply the scalars attached to A and B.
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	buf_alpha = bli_obj_internal_scalar_buffer( &scalar_b );
	buf_beta  = bli_obj_internal_scalar_buffer( c );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Extract from the context the func_t object containing
	// the gemm micro-kernel function addresses, and then query the
	// function address corresponding to the current datatype.
	gemm_ukrs = bli_cntx_get_l3_ukr( BLIS_GEMM_UKR, cntx );
	gemm_ukr  = bli_func_get_dt( dt_exec, gemm_ukrs );

	// Invoke the function.
	f( m,
	   n,
	   k,
	   buf_alpha,
	   buf_a, cs_a, pd_a, ps_a,
	   buf_b, rs_b, pd_b, ps_b,
	   buf_beta,
	   buf_c, rs_c, cs_c,
	   gemm_ukr );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, ukrtype ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   alpha, \
                           void*   a, inc_t cs_a, dim_t pd_a, inc_t ps_a, \
                           void*   b, inc_t rs_b, dim_t pd_b, inc_t ps_b, \
                           void*   beta, \
                           void*   c, inc_t rs_c, inc_t cs_c, \
                           void_fp gemm_ukr  \
                         ) \
{ \
	/* Cast the micro-kernel address to its function pointer type. */ \
	PASTECH(ch,ukrtype) gemm_ukr_cast = gemm_ukr; \
\
	/* Temporary buffer for incremental packing of B. */ \
	ctype           bp[ PASTEMAC(ch,maxkc) * \
	/* !!!! NOTE: This packnr actually needs to be something like maxpacknr
	   if it is to be guaranteed to work in all situations !!!! The right
	   place to define maxpackmr/nr would be in bli_kernel_post_macro_defs.h */ \
	                    PASTEMAC(ch,packnr) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
\
	/* Temporary C buffer for edge cases. */ \
	ctype           ct[ PASTEMAC(ch,maxmr) * \
	                    PASTEMAC(ch,maxnr) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t     rs_ct      = 1; \
	const inc_t     cs_ct      = PASTEMAC(ch,maxmr); \
\
	/* Alias some constants to simpler names. */ \
	const dim_t     MR         = pd_a; \
	const dim_t     NR         = pd_b; \
	const dim_t     PACKNR     = rs_b; \
\
	ctype* restrict one        = PASTEMAC(ch,1); \
	ctype* restrict zero       = PASTEMAC(ch,0); \
	ctype* restrict a_cast     = a; \
	ctype* restrict b_cast     = b; \
	ctype* restrict c_cast     = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
	ctype* restrict b1; \
	ctype* restrict c1; \
	ctype* restrict b2; \
\
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           i, j; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	inc_t           rstep_a; \
	inc_t           cstep_b; \
	inc_t           rstep_c, cstep_c; \
	auxinfo_t       aux; \
\
	/*
	   Assumptions/assertions:
	     rs_a == 1
	     cs_a == PACKMR
	     pd_a == MR
	     ps_a == stride to next micro-panel of A
	     rs_b == PACKNR
	     cs_b == 1
	     pd_b == NR
	     ps_b == stride to next micro-panel of B
	     rs_c == (no assumptions)
	     cs_c == (no assumptions)
	*/ \
\
	/* If any dimension is zero, return immediately. */ \
	if ( bli_zero_dim3( m, n, k ) ) return; \
\
	/* Clear the temporary C buffer in case it has any infs or NaNs. */ \
	PASTEMAC(ch,set0s_mxn)( MR, NR, \
	                        ct, rs_ct, cs_ct ); \
\
	/* Compute number of primary and leftover components of the m and n
	   dimensions. */ \
	n_iter = n / NR; \
	n_left = n % NR; \
\
	m_iter = m / MR; \
	m_left = m % MR; \
\
	if ( n_left ) ++n_iter; \
	if ( m_left ) ++m_iter; \
\
	/* Determine some increments used to step through A, B, and C. */ \
	rstep_a = ps_a; \
\
	cstep_b = ps_b; \
\
	rstep_c = rs_c * MR; \
	cstep_c = cs_c * NR; \
\
	/* Save the panel strides of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_ps_a( ps_a, &aux ); \
	bli_auxinfo_set_ps_b( ps_b, &aux ); \
\
	b1 = b_cast; \
	c1 = c_cast; \
\
	/* Since we pack micro-panels of B incrementaly, one at a time, the
	   address of the next micro-panel of B remains constant. */ \
	b2 = bp; \
\
	/* Save address of next panel of B to the auxinfo_t object. */ \
	bli_auxinfo_set_next_b( b2, &aux ); \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = 0; j < n_iter; ++j ) \
	{ \
		ctype* restrict a1; \
		ctype* restrict c11; \
\
		a1  = a_cast; \
		c11 = c1; \
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* Incrementally pack a single micro-panel of B. */ \
		PASTEMAC(ch,packm_cxk)( BLIS_NO_CONJUGATE, \
		                        n_cur, \
		                        k, \
		                        one, \
		                        b1, 1, rs_b, \
		                        bp,    PACKNR ); \
\
		/* Loop over the m dimension (MR rows at a time). */ \
		for ( i = 0; i < m_iter; ++i ) \
		{ \
			ctype* restrict a2; \
\
			m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
			/* Compute the addresses of the next panels of A and B. */ \
			a2 = a1 + rstep_a; \
			if ( bli_is_last_iter( i, m_iter ) ) \
			{ \
				a2 = a_cast; \
			} \
\
			/* Save address of next panel of A to the auxinfo_t object. */ \
			bli_auxinfo_set_next_a( a2, &aux ); \
\
			/* Handle interior and edge cases separately. */ \
			if ( m_cur == MR && n_cur == NR ) \
			{ \
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr_cast( k, \
				               alpha_cast, \
				               a1, \
				               bp, \
				               beta_cast, \
				               c11, rs_c, cs_c, \
				               &aux ); \
			} \
			else \
			{ \
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr_cast( k, \
				               alpha_cast, \
				               a1, \
				               bp, \
				               zero, \
				               ct, rs_ct, cs_ct, \
				               &aux ); \
\
				/* Scale the bottom edge of C and add the result from above. */ \
				PASTEMAC(ch,xpbys_mxn)( m_cur, n_cur, \
				                        ct,  rs_ct, cs_ct, \
				                        beta_cast, \
				                        c11, rs_c,  cs_c ); \
			} \
\
			a1  += rstep_a; \
			c11 += rstep_c; \
		} \
\
		b1 += cstep_b; \
		c1 += cstep_c; \
	} \
\
/*PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var5: b1", k, NR, b1, NR, 1, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var5: a1", MR, k, a1, 1, MR, "%4.1f", "" );*/ \
}

INSERT_GENTFUNC_BASIC( gemm_ker_var5, gemm_ukr_t )

