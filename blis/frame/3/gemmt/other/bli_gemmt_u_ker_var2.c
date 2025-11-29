/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#define FUNCPTR_T gemmt_fp

typedef void (*FUNCPTR_T)
     (
       doff_t  diagoffc,
       pack_t  schema_a,
       pack_t  schema_b,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       void*   alpha,
       void*   a, inc_t cs_a, inc_t is_a,
                  dim_t pd_a, inc_t ps_a,
       void*   b, inc_t rs_b, inc_t is_b,
                  dim_t pd_b, inc_t ps_b,
       void*   beta,
       void*   c, inc_t rs_c, inc_t cs_c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     );

static FUNCPTR_T GENARRAY(ftypes,gemmt_u_ker_var2);


void bli_gemmt_u_ker_var2
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	num_t     dt_exec   = bli_obj_exec_dt( c );

	doff_t    diagoffc  = bli_obj_diag_offset( c );

	pack_t    schema_a  = bli_obj_pack_schema( a );
	pack_t    schema_b  = bli_obj_pack_schema( b );

	dim_t     m         = bli_obj_length( c );
	dim_t     n         = bli_obj_width( c );
	dim_t     k         = bli_obj_width( a );

	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     cs_a      = bli_obj_col_stride( a );
	inc_t     is_a      = bli_obj_imag_stride( a );
	dim_t     pd_a      = bli_obj_panel_dim( a );
	inc_t     ps_a      = bli_obj_panel_stride( a );

	void*     buf_b     = bli_obj_buffer_at_off( b );
	inc_t     rs_b      = bli_obj_row_stride( b );
	inc_t     is_b      = bli_obj_imag_stride( b );
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

	// Invoke the function.
	f( diagoffc,
	   schema_a,
	   schema_b,
	   m,
	   n,
	   k,
	   buf_alpha,
	   buf_a, cs_a, is_a,
	          pd_a, ps_a,
	   buf_b, rs_b, is_b,
	          pd_b, ps_b,
	   buf_beta,
	   buf_c, rs_c, cs_c,
	   cntx,
	   rntm,
	   thread );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t  diagoffc, \
       pack_t  schema_a, \
       pack_t  schema_b, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       void*   alpha, \
       void*   a, inc_t cs_a, inc_t is_a, \
                  dim_t pd_a, inc_t ps_a, \
       void*   b, inc_t rs_b, inc_t is_b, \
                  dim_t pd_b, inc_t ps_b, \
       void*   beta, \
       void*   c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       thrinfo_t* thread  \
     ) \
{ \
	const num_t     dt         = PASTEMAC(ch,type); \
\
	/* Alias some constants to simpler names. */ \
	const dim_t     MR         = pd_a; \
	const dim_t     NR         = pd_b; \
	/*const dim_t     PACKMR     = cs_a;*/ \
	/*const dim_t     PACKNR     = rs_b;*/ \
\
	/* Query the context for the micro-kernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemm_ukr_ft) \
	                gemm_ukr   = bli_cntx_get_l3_vir_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
\
	/* Temporary C buffer for edge cases. Note that the strides of this
	   temporary buffer are set so that they match the storage of the
	   original C matrix. For example, if C is column-stored, ct will be
	   column-stored as well. */ \
	ctype           ct[ BLIS_STACK_BUF_MAX_SIZE \
	                    / sizeof( ctype ) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const bool      col_pref    = bli_cntx_ukr_prefers_cols_dt( dt, BLIS_GEMM_VIR_UKR, cntx ); \
	const inc_t     rs_ct       = ( col_pref ? 1 : NR ); \
	const inc_t     cs_ct       = ( col_pref ? MR : 1 ); \
\
	ctype* restrict zero       = PASTEMAC(ch,0); \
	ctype* restrict a_cast     = a; \
	ctype* restrict b_cast     = b; \
	ctype* restrict c_cast     = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
	ctype* restrict b1; \
	ctype* restrict c1; \
\
	doff_t          diagoffc_ij; \
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	dim_t           i, j, jp; \
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
	/* Safeguard: If the current panel of C is entirely below the diagonal,
	   it is not stored. So we do nothing. */ \
	if ( bli_is_strictly_below_diag_n( diagoffc, m, n ) ) return; \
\
	/* If there is a zero region to the left of where the diagonal of C
	   intersects the top edge of the panel, adjust the pointer to C and B
	   and treat this case as if the diagonal offset were zero. */ \
	if ( diagoffc > 0 ) \
	{ \
		jp       = diagoffc / NR; \
		j        = jp * NR; \
		n        = n - j; \
		diagoffc = diagoffc % NR; \
		c_cast   = c_cast + (j  )*cs_c; \
		b_cast   = b_cast + (jp )*ps_b; \
	} \
\
	/* If there is a zero region below where the diagonal of C intersects
	   the right edge of the panel, shrink it to prevent "no-op" iterations
	   from executing. */ \
	if ( -diagoffc + n < m ) \
	{ \
		m = -diagoffc + n; \
	} \
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
	/* Save the pack schemas of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_schema_a( schema_a, &aux ); \
	bli_auxinfo_set_schema_b( schema_b, &aux ); \
\
	/* Save the imaginary stride of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_is_a( is_a, &aux ); \
	bli_auxinfo_set_is_b( is_b, &aux ); \
\
	b1 = b_cast; \
	c1 = c_cast; \
\
	thrinfo_t* caucus    = bli_thrinfo_sub_node( thread ); \
	dim_t jr_num_threads = bli_thrinfo_n_way( thread ); \
	dim_t jr_thread_id   = bli_thrinfo_work_id( thread ); \
	dim_t ir_num_threads = bli_thrinfo_n_way( caucus ); \
	dim_t ir_thread_id   = bli_thrinfo_work_id( caucus ); \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = jr_thread_id; j < n_iter; j += jr_num_threads ) \
	{ \
		ctype* restrict a1; \
		ctype* restrict c11; \
		ctype* restrict b2; \
\
		b1 = b_cast + j * cstep_b; \
		c1 = c_cast + j * cstep_c; \
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		b2 = b1; \
\
		/* Interior loop over the m dimension (MR rows at a time). */ \
		for ( i = ir_thread_id; i < m_iter; i += ir_num_threads ) \
		{ \
			ctype* restrict a2; \
\
			a1  = a_cast + i * rstep_a; \
			c11 = c1     + i * rstep_c; \
\
			/* Compute the diagonal offset for the submatrix at (i,j). */ \
			diagoffc_ij = diagoffc - (doff_t)j*NR + (doff_t)i*MR; \
\
			m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
			/* Compute the addresses of the next panels of A and B. */ \
			a2 = bli_gemmt_get_next_a_upanel( caucus, a1, rstep_a ); \
			if ( bli_is_last_iter( i, m_iter, ir_thread_id, ir_num_threads ) ) \
			{ \
				a2 = a_cast; \
				b2 = bli_gemmt_get_next_b_upanel( thread, b1, cstep_b ); \
				if ( bli_is_last_iter( j, n_iter, jr_thread_id, jr_num_threads ) ) \
					b2 = b_cast; \
			} \
\
			/* Save addresses of next panels of A and B to the auxinfo_t
			   object. */ \
			bli_auxinfo_set_next_a( a2, &aux ); \
			bli_auxinfo_set_next_b( b2, &aux ); \
\
			/* If the diagonal intersects the current MR x NR submatrix, we
			   compute it the temporary buffer and then add in the elements
			   on or below the diagonal.
			   Otherwise, if the submatrix is strictly above the diagonal,
			   we compute and store as we normally would.
			   And if we're strictly below the diagonal, we do nothing and
			   continue. */ \
			if ( bli_intersects_diag_n( diagoffc_ij, m_cur, n_cur ) ) \
			{ \
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr \
				( \
				  k, \
				  alpha_cast, \
				  a1, \
				  b1, \
				  zero, \
				  ct, rs_ct, cs_ct, \
				  &aux, \
				  cntx  \
				); \
\
				/* Scale C and add the result to only the stored part. */ \
				PASTEMAC(ch,xpbys_mxn_u)( diagoffc_ij, \
				                          m_cur, n_cur, \
				                          ct,  rs_ct, cs_ct, \
				                          beta_cast, \
				                          c11, rs_c,  cs_c ); \
			} \
			else if ( bli_is_strictly_above_diag_n( diagoffc_ij, m_cur, n_cur ) ) \
			{ \
				/* Handle interior and edge cases separately. */ \
				if ( m_cur == MR && n_cur == NR ) \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					gemm_ukr \
					( \
					  k, \
					  alpha_cast, \
					  a1, \
					  b1, \
					  beta_cast, \
					  c11, rs_c, cs_c, \
					  &aux, \
					  cntx  \
					); \
				} \
				else \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					gemm_ukr \
					( \
					  k, \
					  alpha_cast, \
					  a1, \
					  b1, \
					  zero, \
					  ct, rs_ct, cs_ct, \
					  &aux, \
					  cntx  \
					); \
\
					/* Scale the edge of C and add the result. */ \
					PASTEMAC(ch,xpbys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        beta_cast, \
					                        c11, rs_c,  cs_c ); \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( gemmt_u_ker_var2 )

