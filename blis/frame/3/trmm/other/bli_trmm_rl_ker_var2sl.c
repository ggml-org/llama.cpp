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

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)
     (
       doff_t  diagoffb,
       pack_t  schema_a,
       pack_t  schema_b,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       void*   alpha,
       void*   a, inc_t cs_a, dim_t pd_a, inc_t ps_a,
       void*   b, inc_t rs_b, dim_t pd_b, inc_t ps_b,
       void*   beta,
       void*   c, inc_t rs_c, inc_t cs_c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     );

static FUNCPTR_T GENARRAY(ftypes,trmm_rl_ker_var2sl);

//
// -- Macrokernel functions for slab partitioning ------------------------------
//

void bli_trmm_rl_ker_var2sl
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

	doff_t    diagoffb  = bli_obj_diag_offset( b );

	pack_t    schema_a  = bli_obj_pack_schema( a );
	pack_t    schema_b  = bli_obj_pack_schema( b );

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
	f( diagoffb,
	   schema_a,
	   schema_b,
	   m,
	   n,
	   k,
	   buf_alpha,
	   buf_a, cs_a, pd_a, ps_a,
	   buf_b, rs_b, pd_b, ps_b,
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
       doff_t  diagoffb, \
       pack_t  schema_a, \
       pack_t  schema_b, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       void*   alpha, \
       void*   a, inc_t cs_a, dim_t pd_a, inc_t ps_a, \
       void*   b, inc_t rs_b, dim_t pd_b, inc_t ps_b, \
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
	const dim_t     PACKMR     = cs_a; \
	const dim_t     PACKNR     = rs_b; \
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
	ctype* restrict one        = PASTEMAC(ch,1); \
	ctype* restrict zero       = PASTEMAC(ch,0); \
	ctype* restrict a_cast     = a; \
	ctype* restrict b_cast     = b; \
	ctype* restrict c_cast     = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
	ctype* restrict b1; \
	ctype* restrict c1; \
\
	doff_t          diagoffb_j; \
	dim_t           k_full; \
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	dim_t           k_b1121; \
	dim_t           off_b1121; \
	dim_t           i, j; \
	inc_t           rstep_a; \
	inc_t           cstep_b; \
	inc_t           rstep_c, cstep_c; \
	inc_t           istep_a; \
	inc_t           istep_b; \
	inc_t           off_scl; \
	inc_t           ss_b_num; \
	inc_t           ss_b_den; \
	inc_t           ps_b_cur; \
	inc_t           is_b_cur; \
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
	/* Safety trap: Certain indexing within this macro-kernel does not
	   work as intended if both MR and NR are odd. */ \
	if ( ( bli_is_odd( PACKMR ) && bli_is_odd( NR ) ) || \
	     ( bli_is_odd( PACKNR ) && bli_is_odd( MR ) ) ) bli_abort(); \
\
	/* If any dimension is zero, return immediately. */ \
	if ( bli_zero_dim3( m, n, k ) ) return; \
\
	/* Safeguard: If the current panel of B is entirely above the diagonal,
	   it is implicitly zero. So we do nothing. */ \
	if ( bli_is_strictly_above_diag_n( diagoffb, k, n ) ) return; \
\
	/* Compute k_full. For all trmm, k_full is simply k. This is
	   needed because some parameter combinations of trmm reduce k
	   to advance past zero regions in the triangular matrix, and
	   when computing the imaginary stride of A (the non-triangular
	   matrix), which is used by 4m1/3m1 implementations, we need
	   this unreduced value of k. */ \
	k_full = k; \
\
	/* Compute indexing scaling factor for for 4m or 3m. This is
	   needed because one of the packing register blocksizes (PACKMR
	   or PACKNR) is used to index into the micro-panels of the non-
	   triangular matrix when computing with a diagonal-intersecting
	   micro-panel of the triangular matrix. In the case of 4m or 3m,
	   real values are stored in both sub-panels, and so the indexing
	   needs to occur in units of real values. The value computed
	   here is divided into the complex pointer offset to cause the
	   pointer to be advanced by the correct value. */ \
	if ( bli_is_4mi_packed( schema_b ) || \
	     bli_is_3mi_packed( schema_b ) || \
	     bli_is_rih_packed( schema_b ) ) off_scl = 2; \
	else                                 off_scl = 1; \
\
	/* Compute the storage stride scaling. Usually this is just 1.
	   However, in the case of interleaved 3m, we need to scale the
	   offset by 3/2. And if we are packing real-only, imag-only, or
	   summed-only, we need to scale the computed panel sizes by 1/2
	   to compensate for the fact that the pointer arithmetic occurs
	   in terms of complex elements rather than real elements. */ \
	if      ( bli_is_3mi_packed( schema_b ) ) { ss_b_num = 3; ss_b_den = 2; } \
	else if ( bli_is_rih_packed( schema_b ) ) { ss_b_num = 1; ss_b_den = 2; } \
	else                                      { ss_b_num = 1; ss_b_den = 1; } \
\
	/* If there is a zero region above where the diagonal of B intersects
	   the left edge of the panel, adjust the pointer to A and treat this
	   case as if the diagonal offset were zero. Note that we don't need to
	   adjust the pointer to B since packm would have simply skipped over
	   the region that was not stored. */ \
	if ( diagoffb < 0 ) \
	{ \
		j        = -diagoffb; \
		k        = k - j; \
		diagoffb = 0; \
		a_cast   = a_cast + ( j * PACKMR ) / off_scl; \
	} \
\
	/* If there is a zero region to the right of where the diagonal
	   of B intersects the bottom of the panel, shrink it to prevent
	   "no-op" iterations from executing. */ \
	if ( diagoffb + k < n ) \
	{ \
		n = diagoffb + k; \
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
	istep_a = PACKMR * k_full; \
	istep_b = PACKNR * k; \
\
	if ( bli_is_odd( istep_a ) ) istep_a += 1; \
	if ( bli_is_odd( istep_b ) ) istep_b += 1; \
\
	/* Save the pack schemas of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_schema_a( schema_a, &aux ); \
	bli_auxinfo_set_schema_b( schema_b, &aux ); \
\
	/* Save the imaginary stride of A to the auxinfo_t object. */ \
	bli_auxinfo_set_is_a( istep_a, &aux ); \
\
	thrinfo_t* caucus = bli_thrinfo_sub_node( thread ); \
\
	dim_t jr_nt  = bli_thrinfo_n_way( thread ); \
	dim_t jr_tid = bli_thrinfo_work_id( thread ); \
	dim_t ir_nt  = bli_thrinfo_n_way( caucus ); \
	dim_t ir_tid = bli_thrinfo_work_id( caucus ); \
\
	dim_t jr_start, jr_end; \
	dim_t ir_start, ir_end; \
	dim_t jr_inc,   ir_inc; \
\
	/* Note that we partition the 2nd loop into two regions: the rectangular
	   part of B, and the triangular portion. */ \
	dim_t n_iter_rct; \
	dim_t n_iter_tri; \
\
	if ( bli_is_strictly_below_diag_n( diagoffb, m, n ) ) \
	{ \
		/* If the entire panel of B does not intersect the diagonal, there is
		   no triangular region, and therefore we can skip the second set of
		   loops. */ \
		n_iter_rct = n_iter; \
		n_iter_tri = 0; \
	} \
	else \
	{ \
		/* If the panel of B does intersect the diagonal, compute the number of
		   iterations in the rectangular region by dividing NR into the diagonal
		   offset. (There should never be any remainder in this division.) The
		   number of iterations in the triangular (or trapezoidal) region is
		   computed as the remaining number of iterations in the n dimension. */ \
		n_iter_rct = diagoffb / NR; \
		n_iter_tri = n_iter - n_iter_rct; \
	} \
\
	/* Use slab assignment of micropanels to threads in the 2nd and 1st
	   loops for the initial rectangular region of B (if it exists). */ \
	bli_thread_range_jrir_sl( thread, n_iter_rct, 1, FALSE, &jr_start, &jr_end, &jr_inc ); \
	bli_thread_range_jrir_sl( caucus, m_iter,     1, FALSE, &ir_start, &ir_end, &ir_inc ); \
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
		{ \
			/* Save the 4m1/3m1 imaginary stride of B to the auxinfo_t
			   object. */ \
			bli_auxinfo_set_is_b( istep_b, &aux ); \
\
			/* Loop over the m dimension (MR rows at a time). */ \
			for ( i = ir_start; i < ir_end; i += ir_inc ) \
			{ \
				ctype* restrict a2; \
\
				a1  = a_cast + i * rstep_a; \
				c11 = c1     + i * rstep_c; \
\
				m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
				/* Compute the addresses of the next panels of A and B. */ \
				a2 = bli_trmm_get_next_a_upanel( a1, rstep_a, ir_inc ); \
				if ( bli_is_last_iter_sl( i, m_iter, ir_tid, ir_nt ) ) \
				{ \
					a2 = a_cast; \
					b2 = bli_trmm_get_next_b_upanel( b1, cstep_b, jr_inc ); \
					if ( bli_is_last_iter_sl( j, n_iter, jr_tid, jr_nt ) ) \
						b2 = b_cast; \
				} \
\
				/* Save addresses of next panels of A and B to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_next_a( a2, &aux ); \
				bli_auxinfo_set_next_b( b2, &aux ); \
\
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
					  one, \
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
					/* Add the result to the edge of C. */ \
					PASTEMAC(ch,adds_mxn)( m_cur, n_cur, \
					                       ct,  rs_ct, cs_ct, \
					                       c11, rs_c,  cs_c ); \
				} \
			} \
		} \
	} \
\
	/* If there is no triangular region, then we're done. */ \
	if ( n_iter_tri == 0 ) return; \
\
	/* Use round-robin assignment of micropanels to threads in the 2nd loop
	   for the remaining triangular region of B (if it exists).
	   NOTE: We don't need to call bli_thread_range_jrir*() here since we
	   employ a hack that calls for each thread to execute every iteration
	   of the jr and ir loops but skip all but the pointer increment for
	   iterations that are not assigned to it. */ \
\
	/* Advance the starting b1 and c1 pointers to the positions corresponding
	   to the start of the triangular region of B. */ \
	jr_start = n_iter_rct; \
	b1 = b_cast + jr_start * cstep_b; \
	c1 = c_cast + jr_start * cstep_c; \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = jr_start; j < n_iter; ++j ) \
	{ \
		ctype* restrict a1; \
		ctype* restrict c11; \
		ctype* restrict b2; \
\
		diagoffb_j = diagoffb - ( doff_t )j*NR; \
\
		/* Determine the offset to the beginning of the panel that
		   was packed so we can index into the corresponding location
		   in A. Then compute the length of that panel. */ \
		off_b1121 = bli_max( -diagoffb_j, 0 ); \
		k_b1121   = k - off_b1121; \
\
		a1  = a_cast; \
		c11 = c1; \
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		b2 = b1; \
\
		/* If the current panel of B intersects the diagonal, scale C
		   by beta. If it is strictly below the diagonal, scale by one.
		   This allows the current macro-kernel to work for both trmm
		   and trmm3. */ \
		{ \
			/* Compute the panel stride for the current diagonal-
			   intersecting micro-panel. */ \
			is_b_cur  = k_b1121 * PACKNR; \
			is_b_cur += ( bli_is_odd( is_b_cur ) ? 1 : 0 ); \
			ps_b_cur  = ( is_b_cur * ss_b_num ) / ss_b_den; \
\
			if ( bli_trmm_my_iter( j, thread ) ) { \
\
			/* Save the 4m1/3m1 imaginary stride of B to the auxinfo_t
			   object. */ \
			bli_auxinfo_set_is_b( is_b_cur, &aux ); \
\
			/* Loop over the m dimension (MR rows at a time). */ \
			for ( i = 0; i < m_iter; ++i ) \
			{ \
				if ( bli_trmm_my_iter( i, caucus ) ) { \
\
				ctype* restrict a1_i; \
				ctype* restrict a2; \
\
				m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
				a1_i = a1 + ( off_b1121 * PACKMR ) / off_scl; \
\
				/* Compute the addresses of the next panels of A and B. */ \
				a2 = a1; \
				if ( bli_is_last_iter_rr( i, m_iter, 0, 1 ) ) \
				{ \
					a2 = a_cast; \
					b2 = b1; \
					if ( bli_is_last_iter_rr( j, n_iter, jr_tid, jr_nt ) ) \
						b2 = b_cast; \
				} \
\
				/* Save addresses of next panels of A and B to the auxinfo_t
				   object. */ \
				bli_auxinfo_set_next_a( a2, &aux ); \
				bli_auxinfo_set_next_b( b2, &aux ); \
\
				/* Handle interior and edge cases separately. */ \
				if ( m_cur == MR && n_cur == NR ) \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					gemm_ukr \
					( \
					  k_b1121, \
					  alpha_cast, \
					  a1_i, \
					  b1, \
					  beta_cast, \
					  c11, rs_c, cs_c, \
					  &aux, \
					  cntx  \
					); \
				} \
				else \
				{ \
					/* Copy edge elements of C to the temporary buffer. */ \
					PASTEMAC(ch,copys_mxn)( m_cur, n_cur, \
					                        c11, rs_c,  cs_c, \
					                        ct,  rs_ct, cs_ct ); \
\
					/* Invoke the gemm micro-kernel. */ \
					gemm_ukr \
					( \
					  k_b1121, \
					  alpha_cast, \
					  a1_i, \
					  b1, \
					  beta_cast, \
					  ct, rs_ct, cs_ct, \
					  &aux, \
					  cntx  \
					); \
\
					/* Copy the result to the edge of C. */ \
					PASTEMAC(ch,copys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        c11, rs_c,  cs_c ); \
				} \
				} \
\
				a1  += rstep_a; \
				c11 += rstep_c; \
			} \
			} \
\
			b1 += ps_b_cur; \
		} \
\
		c1 += cstep_c; \
	} \
\
/*PASTEMAC(ch,fprintm)( stdout, "trmm_rl_ker_var2sl: a1", MR, k_b1121, a1, 1, MR, "%4.1f", "" );*/ \
/*PASTEMAC(ch,fprintm)( stdout, "trmm_rl_ker_var2sl: b1", k_b1121, NR, b1_i, NR, 1, "%4.1f", "" );*/ \
}

INSERT_GENTFUNC_BASIC0( trmm_rl_ker_var2sl )

