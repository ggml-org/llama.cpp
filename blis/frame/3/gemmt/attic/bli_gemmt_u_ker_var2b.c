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

static FUNCPTR_T GENARRAY(ftypes,gemmt_u_ker_var2b);


void bli_gemmt_u_ker_var2b
     (
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl,
             thrinfo_t* thread
     )
{
	const num_t  dt_exec   = bli_obj_exec_dt( c );

	const doff_t diagoffc  = bli_obj_diag_offset( c );

	const pack_t schema_a  = bli_obj_pack_schema( a );
	const pack_t schema_b  = bli_obj_pack_schema( b );

	const dim_t  m         = bli_obj_length( c );
	const dim_t  n         = bli_obj_width( c );
	const dim_t  k         = bli_obj_width( a );

	const void*  buf_a     = bli_obj_buffer_at_off( a );
	const inc_t  cs_a      = bli_obj_col_stride( a );
	const inc_t  is_a      = bli_obj_imag_stride( a );
	const dim_t  pd_a      = bli_obj_panel_dim( a );
	const inc_t  ps_a      = bli_obj_panel_stride( a );

	const void*  buf_b     = bli_obj_buffer_at_off( b );
	const inc_t  rs_b      = bli_obj_row_stride( b );
	const inc_t  is_b      = bli_obj_imag_stride( b );
	const dim_t  pd_b      = bli_obj_panel_dim( b );
	const inc_t  ps_b      = bli_obj_panel_stride( b );

	      void*  buf_c     = bli_obj_buffer_at_off( c );
	const inc_t  rs_c      = bli_obj_row_stride( c );
	const inc_t  cs_c      = bli_obj_col_stride( c );

	// Detach and multiply the scalars attached to A and B.
	obj_t  scalar_a, scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	const void* buf_alpha = bli_obj_internal_scalar_buffer( &scalar_b );
	const void* buf_beta  = bli_obj_internal_scalar_buffer( c );

	// Index into the type combination array to extract the correct
	// function pointer.
	ftypes[dt_exec]
	(
	  diagoffc,
	  schema_a,
	  schema_b,
	  m,
	  n,
	  k,
	  ( void* )buf_alpha,
	  ( void* )buf_a, cs_a, is_a,
	                  pd_a, ps_a,
	  ( void* )buf_b, rs_b, is_b,
	                  pd_b, ps_b,
	  ( void* )buf_beta,
	           buf_c, rs_c, cs_c,
	  ( cntx_t* )cntx,
	  rntm,
	  thread
	);
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
\
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
	   and treat this case as if the diagonal offset were zero.
	   NOTE: It's possible that after this pruning that the diagonal offset
	   is still positive (though it is guaranteed to be less than NR). */ \
	if ( diagoffc > 0 ) \
	{ \
		const dim_t jp = diagoffc / NR; \
		const dim_t j  = jp * NR; \
\
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
	const dim_t n_iter = n / NR + ( n % NR ? 1 : 0 ); \
	const dim_t n_left = n % NR; \
\
	const dim_t m_iter = m / MR + ( m % MR ? 1 : 0 ); \
	const dim_t m_left = m % MR; \
\
	/* Determine some increments used to step through A, B, and C. */ \
	const inc_t rstep_a = ps_a; \
\
	const inc_t cstep_b = ps_b; \
\
	const inc_t rstep_c = rs_c * MR; \
	const inc_t cstep_c = cs_c * NR; \
\
	/* Save the pack schemas of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_schema_a( schema_a, &aux ); \
	bli_auxinfo_set_schema_b( schema_b, &aux ); \
\
	/* Save the imaginary stride of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_is_a( is_a, &aux ); \
	bli_auxinfo_set_is_b( is_b, &aux ); \
\
	/* Save the virtual microkernel address and the params. */ \
	/*bli_auxinfo_set_ukr( gemm_ukr, &aux );*/ \
	/*bli_auxinfo_set_params( params, &aux );*/ \
\
	/* Save the desired output datatype (indicating no typecasting). */ \
	/*bli_auxinfo_set_dt_on_output( dt, &aux );*/ \
\
	const dim_t jr_inc = 1; \
	const dim_t ir_inc = 1; \
\
	/* Determine the starting microtile offsets and number of microtiles to
	   compute for each thread. Note that assignment of microtiles is done
	   according to the tlb policy. */ \
	dim_t jr_st, ir_st; \
	const dim_t n_ut_for_me \
	= \
	bli_thread_range_tlb( thread, diagoffc, BLIS_UPPER, m, n, MR, NR, \
	                      &jr_st, &ir_st ); \
\
	/* It's possible that there are so few microtiles relative to the number
	   of threads that one or more threads gets no work. If that happens, those
	   threads can return early. */ \
	if ( n_ut_for_me == 0 ) return; \
\
	/* Start the jr/ir loops with the current thread's microtile offsets computed
	   by bli_thread_range_tlb(). */ \
	dim_t i = ir_st; \
	dim_t j = jr_st; \
\
	/* Initialize a counter to track the number of microtiles computed by the
	   current thread. */ \
	dim_t ut = 0; \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( ; true; ++j ) \
	{ \
		ctype* restrict b1 = b_cast + j * cstep_b; \
		ctype* restrict c1 = c_cast + j * cstep_c; \
\
		/* Compute the diagonal offset for the column of microtiles at (0,j). */ \
		const doff_t diagoffc_j = diagoffc - (doff_t)j*NR; \
		const dim_t  n_cur      = ( bli_is_not_edge_f( j, n_iter, n_left ) \
		                            ? NR : n_left ); \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		ctype* restrict b2 = b1; \
\
		/* Interior loop over the m dimension (MR rows at a time). */ \
		for ( ; i < m_iter; ++i ) \
		{ \
			/* Compute the diagonal offset for the microtile at (i,j). */ \
			const doff_t diagoffc_ij = diagoffc_j + (doff_t)i*MR; \
			const dim_t  m_cur       = ( bli_is_not_edge_f( i, m_iter, m_left ) \
			                             ? MR : m_left ); \
\
			/* If the diagonal intersects the current MR x NR submatrix, we
			   compute it the temporary buffer and then add in the elements
			   on or below the diagonal.
			   Otherwise, if the submatrix is strictly above the diagonal,
			   we compute and store as we normally would.
			   And if we're strictly above the diagonal, we simply advance
			   to last microtile before the bottom of the matrix. */ \
			if ( bli_intersects_diag_n( diagoffc_ij, m_cur, n_cur ) ) \
			{ \
				ctype* restrict a1  = a_cast + i * rstep_a; \
				ctype* restrict c11 = c1     + i * rstep_c; \
\
				/* Compute the addresses of the next panels of A and B. */ \
				ctype* restrict a2 \
				= bli_gemmt_get_next_a_upanel( a1, rstep_a, ir_inc ); \
				if ( bli_is_last_iter_tlb_u( diagoffc_ij, MR, NR ) ) \
				{ \
					a2 = bli_gemmt_u_wrap_a_upanel( a_cast, rstep_a, \
					                                diagoffc_j, MR, NR ); \
					b2 = bli_gemmt_get_next_b_upanel( b1, cstep_b, jr_inc ); \
					/* We don't bother computing b2 for the last iteration of the
					   jr loop since the current thread won't know its j_st until
					   the next time it calls bli_thread_range_tlb(). */ \
				} \
\
				/* Save addresses of next panels of A and B to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_next_a( a2, &aux ); \
				bli_auxinfo_set_next_b( b2, &aux ); \
\
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr \
				( \
				  MR, \
				  NR, \
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
\
				ut += 1; \
				if ( ut == n_ut_for_me ) return; \
			} \
			else if ( bli_is_strictly_above_diag_n( diagoffc_ij, m_cur, n_cur ) ) \
			{ \
				ctype* restrict a1  = a_cast + i * rstep_a; \
				ctype* restrict c11 = c1     + i * rstep_c; \
\
				/* Compute the addresses of the next panels of A and B. */ \
				ctype* restrict a2 \
				= bli_gemmt_get_next_a_upanel( a1, rstep_a, ir_inc ); \
\
				/* Save addresses of next panels of A and B to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_next_a( a2, &aux ); \
				bli_auxinfo_set_next_b( b2, &aux ); \
\
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr \
				( \
				  m_cur, \
				  n_cur, \
				  k, \
				  alpha_cast, \
				  a1, \
				  b1, \
				  beta_cast, \
				  c11, rs_c, cs_c, \
				  &aux, \
				  cntx  \
				); \
\
				ut += 1; \
				if ( ut == n_ut_for_me ) return; \
			} \
			else /* if ( bli_is_strictly_below_diag_n( diagoffc_ij, m_cur, n_cur ) ) */ \
			{ \
				/* Skip past the microtiles strictly below the diagonal. */ \
				i = m_iter - 1; \
			} \
		} \
\
		i = 0; \
	} \
}

INSERT_GENTFUNC_BASIC0( gemmt_u_ker_var2b )

