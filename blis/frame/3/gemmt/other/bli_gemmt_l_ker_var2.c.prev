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

static FUNCPTR_T GENARRAY(ftypes,gemmt_l_ker_var2);


void bli_gemmt_l_ker_var2
     (
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  c,
       const cntx_t* cntx,
             rntm_t* rntm,
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
	ctype* restrict b1; \
	ctype* restrict c1; \
\
	doff_t          diagoffc_ij; \
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	dim_t           i, j, ip; \
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
	/* Safeguard: If the current panel of C is entirely above the diagonal,
	   it is not stored. So we do nothing. */ \
	if ( bli_is_strictly_above_diag_n( diagoffc, m, n ) ) return; \
\
	/* If there is a zero region above where the diagonal of C intersects
	   the left edge of the panel, adjust the pointer to C and A and treat
	   this case as if the diagonal offset were zero. */ \
	if ( diagoffc < 0 ) \
	{ \
		ip       = -diagoffc / MR; \
		i        = ip * MR; \
		m        = m - i; \
		diagoffc = -diagoffc % MR; \
		c_cast   = c_cast + (i  )*rs_c; \
		a_cast   = a_cast + (ip )*ps_a; \
	} \
\
	/* If there is a zero region to the right of where the diagonal
	   of C intersects the bottom of the panel, shrink it to prevent
	   "no-op" iterations from executing. */ \
	if ( diagoffc + m < n ) \
	{ \
		n = diagoffc + m; \
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
	/* Save the desired output datatype (indicating no typecasting). */ \
	/*bli_auxinfo_set_dt_on_output( dt, &aux );*/ \
\
	/* The 'thread' argument points to the thrinfo_t node for the 2nd (jr)
	   loop around the microkernel. Here we query the thrinfo_t node for the
	   1st (ir) loop around the microkernel. */ \
	thrinfo_t* caucus = bli_thrinfo_sub_node( thread ); \
\
	/* Query the number of threads and thread ids for each loop. */ \
	dim_t jr_nt  = bli_thread_n_way( thread ); \
	dim_t jr_tid = bli_thread_work_id( thread ); \
	dim_t ir_nt  = bli_thread_n_way( caucus ); \
	dim_t ir_tid = bli_thread_work_id( caucus ); \
\
	dim_t jr_start, jr_end; \
	dim_t ir_start, ir_end; \
	dim_t jr_inc,   ir_inc; \
\
	/* Note that we partition the 2nd loop into two regions: the rectangular
	   part of C, and the triangular portion. */ \
	dim_t n_iter_rct; \
	dim_t n_iter_tri; \
\
	if ( bli_is_strictly_below_diag_n( diagoffc, m, n ) ) \
	{ \
		/* If the entire panel of C does not intersect the diagonal, there is
		   no triangular region, and therefore we can skip the second set of
		   loops. */ \
		n_iter_rct = n_iter; \
		n_iter_tri = 0; \
	} \
	else \
	{ \
		/* If the panel of C does intersect the diagonal, compute the number of
		   iterations in the rectangular region by dividing NR into the diagonal
		   offset. Any remainder from this integer division is discarded, which
		   is what we want. That is, we want the rectangular region to contain
		   as many columns of whole microtiles as possible without including any
		   microtiles that intersect the diagonal. The number of iterations in
		   the triangular (or trapezoidal) region is computed as the remaining
		   number of iterations in the n dimension. */ \
		n_iter_rct = diagoffc / NR; \
		n_iter_tri = n_iter - n_iter_rct; \
	} \
\
	/* Determine the thread range and increment for the 2nd and 1st loops for
	   the initial rectangular region of C (if it exists).
	   NOTE: The definition of bli_thread_range_jrir() will depend on whether
	   slab or round-robin partitioning was requested at configure-time. */ \
	bli_thread_range_jrir( thread, n_iter_rct, 1, FALSE, &jr_start, &jr_end, &jr_inc ); \
	bli_thread_range_jrir( caucus, m_iter,     1, FALSE, &ir_start, &ir_end, &ir_inc ); \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = jr_start; j < jr_end; j += jr_inc ) \
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
		for ( i = ir_start; i < ir_end; i += ir_inc ) \
		{ \
			ctype* restrict a2; \
\
			a1  = a_cast + i * rstep_a; \
			c11 = c1     + i * rstep_c; \
\
			/* No need to compute the diagonal offset for the rectangular
			   region. */ \
			/*diagoffc_ij = diagoffc - (doff_t)j*NR + (doff_t)i*MR;*/ \
\
			m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
			/* Compute the addresses of the next panels of A and B. */ \
			a2 = bli_gemmt_get_next_a_upanel( a1, rstep_a, ir_inc ); \
			if ( bli_is_last_iter( i, m_iter, ir_tid, ir_nt ) ) \
			{ \
				a2 = a_cast; \
				b2 = bli_gemmt_get_next_b_upanel( b1, cstep_b, jr_inc ); \
				if ( bli_is_last_iter( j, n_iter, jr_tid, jr_nt ) ) \
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
			   Otherwise, if the submatrix is strictly below the diagonal,
			   we compute and store as we normally would.
			   And if we're strictly above the diagonal, we do nothing and
			   continue. */ \
			{ \
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
			} \
		} \
	} \
\
	/* If there is no triangular region, then we're done. */ \
	if ( n_iter_tri == 0 ) return; \
\
	/* Use round-robin assignment of micropanels to threads in the 2nd loop
	   and the default (slab or rr) partitioning in the 1st loop for the
	   remaining triangular region of C. */ \
	bli_thread_range_jrir_rr( thread, n_iter_tri, 1, FALSE, &jr_start, &jr_end, &jr_inc ); \
\
	/* Advance the start and end iteration offsets for the triangular region
	   by the number of iterations used for the rectangular region. */ \
	jr_start += n_iter_rct; \
	jr_end   += n_iter_rct; \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = jr_start; j < jr_end; j += jr_inc ) \
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
		for ( i = ir_start; i < ir_end; i += ir_inc ) \
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
			a2 = bli_gemmt_get_next_a_upanel( a1, rstep_a, ir_inc ); \
			if ( bli_is_last_iter( i, m_iter, ir_tid, ir_nt ) ) \
			{ \
				a2 = a_cast; \
				b2 = bli_gemmt_get_next_b_upanel( b1, cstep_b, jr_inc ); \
				if ( bli_is_last_iter_rr( j, n_iter, jr_tid, jr_nt ) ) \
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
			   Otherwise, if the submatrix is strictly below the diagonal,
			   we compute and store as we normally would.
			   And if we're strictly above the diagonal, we do nothing and
			   continue. */ \
			if ( bli_intersects_diag_n( diagoffc_ij, m_cur, n_cur ) ) \
			{ \
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
				PASTEMAC(ch,xpbys_mxn_l)( diagoffc_ij, \
				                          m_cur, n_cur, \
				                          ct,  rs_ct, cs_ct, \
				                          beta_cast, \
				                          c11, rs_c,  cs_c ); \
			} \
			else if ( bli_is_strictly_below_diag_n( diagoffc_ij, m_cur, n_cur ) ) \
			{ \
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
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( gemmt_l_ker_var2 )

