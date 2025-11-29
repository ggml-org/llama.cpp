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

void bli_trsm_lu_ker_var2
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	const num_t  dt_comp   = bli_gemm_var_cntl_comp_dt( cntl );
	const num_t  dt_a      = bli_obj_dt( a );
	const num_t  dt_b      = bli_obj_dt( b );
	const num_t  dt_c      = bli_obj_dt( c );

	const siz_t  dt_a_size = bli_dt_size( dt_a );
	const siz_t  dt_b_size = bli_dt_size( dt_b );
	const siz_t  dt_c_size = bli_dt_size( dt_c );

	      doff_t diagoffa  = bli_obj_diag_offset( a );

	const pack_t schema_a  = bli_obj_pack_schema( a );
	const pack_t schema_b  = bli_obj_pack_schema( b );

	      dim_t  m         = bli_obj_length( c );
	      dim_t  n         = bli_obj_width( c );
	      dim_t  k         = bli_obj_width( a );

	const void*  buf_a     = bli_obj_buffer_at_off( a );
	const inc_t  cs_a      = bli_obj_col_stride( a );
	const dim_t  pd_a      = bli_obj_panel_dim( a );
	const inc_t  ps_a      = bli_obj_panel_stride( a );

	const void*  buf_b     = bli_obj_buffer_at_off( b );
	const inc_t  rs_b      = bli_obj_row_stride( b );
	const dim_t  pd_b      = bli_obj_panel_dim( b );
	const inc_t  ps_b      = bli_obj_panel_stride( b );

	      void*  buf_c     = bli_obj_buffer_at_off( c );
	const inc_t  rs_c      = bli_obj_row_stride( c );
	const inc_t  cs_c      = bli_obj_col_stride( c );

	// Grab the address of the internal scalar buffer for the scalar
	// attached to B (the non-triangular matrix). This will be the alpha
	// scalar used in the gemmtrsm subproblems (ie: the scalar that would
	// be applied to the packed copy of B prior to it being updated by
	// the trsm subproblem). This scalar may be unit, if for example it
	// was applied during packing.
	const void* buf_alpha1 = bli_obj_internal_scalar_buffer( b );

	// Grab the address of the internal scalar buffer for the scalar
	// attached to C. This will be the "beta" scalar used in the gemm-only
	// subproblems that correspond to micro-panels that do not intersect
	// the diagonal. We need this separate scalar because it's possible
	// that the alpha attached to B was reset, if it was applied during
	// packing.
	const void* buf_alpha2 = bli_obj_internal_scalar_buffer( c );

	// Alias some constants to simpler names.
	const dim_t     MR          = pd_a;
	const dim_t     NR          = pd_b;
	const dim_t     PACKMR      = cs_a;
	const dim_t     PACKNR      = rs_b;

	// Cast the micro-kernel address to its function pointer type.
	gemmtrsm_ukr_ft gemmtrsm_ukr = bli_trsm_var_cntl_gemmtrsm_ukr( cntl );
	gemm_ukr_ft     gemm_ukr     = bli_trsm_var_cntl_gemm_ukr( cntl );
	const void*     params       = bli_trsm_var_cntl_params( cntl );

	const void* minus_one   = bli_obj_buffer_for_const( dt_comp, &BLIS_MINUS_ONE );
	const char* a_cast      = buf_a;
	const char* b_cast      = buf_b;
	      char* c_cast      = buf_c;
	const char* alpha1_cast = buf_alpha1;
	const char* alpha2_cast = buf_alpha2;

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
	*/

	// Safety trap: Certain indexing within this macro-kernel does not
	// work as intended if both MR and NR are odd.
	if ( ( bli_is_odd( PACKMR ) && bli_is_odd( NR ) ) ||
	     ( bli_is_odd( PACKNR ) && bli_is_odd( MR ) ) ) bli_abort();

	// If any dimension is zero, return immediately.
	if ( bli_zero_dim3( m, n, k ) ) return;

	// Safeguard: If matrix A is below the diagonal, it is implicitly zero.
	// So we do nothing.
	if ( bli_is_strictly_below_diag_n( diagoffa, m, k ) ) return;

	// If there is a zero region to the left of where the diagonal of A
	// intersects the top edge of the block, adjust the pointer to B and
	// treat this case as if the diagonal offset were zero. Note that we
	// don't need to adjust the pointer to A since packm would have simply
	// skipped over the region that was not stored.
	if ( diagoffa > 0 )
	{
		k        -= diagoffa;
		b_cast   += diagoffa * PACKNR * dt_b_size;
		diagoffa  = 0;
	}

	// If there is a zero region below where the diagonal of A intersects the
	// right side of the block, shrink it to prevent "no-op" iterations from
	// executing.
	if ( -diagoffa + k < m )
	{
		m = -diagoffa + k;
	}

	// Check the k dimension, which needs to be a multiple of MR. If k
	// isn't a multiple of MR, we adjust it higher to satisfy the micro-
	// kernel, which is expecting to perform an MR x MR triangular solve.
	// This adjustment of k is consistent with what happened when A was
	// packed: all of its bottom/right edges were zero-padded, and
	// furthermore, the panel that stores the bottom-right corner of the
	// matrix has its diagonal extended into the zero-padded region (as
	// identity). This allows the trsm of that bottom-right panel to
	// proceed without producing any infs or NaNs that would infect the
	// "good" values of the corresponding block of B.
	if ( k % MR != 0 ) k += MR - ( k % MR );

	// NOTE: We don't need to check that m is a multiple of PACKMR since we
	// know that the underlying buffer was already allocated to have an m
	// dimension that is a multiple of PACKMR, with the region between the
	// last row and the next multiple of MR zero-padded accordingly.

	// Compute number of primary and leftover components of the m and n
	// dimensions.
	const dim_t n_iter = n / NR + ( n % NR ? 1 : 0 );
	const dim_t n_left = n % NR;

	const dim_t m_iter = m / MR + ( m % MR ? 1 : 0 );
	const dim_t m_left = m % MR;

	// Determine some increments used to step through A, B, and C.
	const inc_t rstep_a = ps_a * dt_a_size;

	const inc_t cstep_b = ps_b * dt_b_size;

	const inc_t rstep_c = rs_c * MR * dt_c_size;
	const inc_t cstep_c = cs_c * NR * dt_c_size;

	auxinfo_t aux;

	// Save the pack schemas of A and B to the auxinfo_t object.
	bli_auxinfo_set_schema_a( schema_a, &aux );
	bli_auxinfo_set_schema_b( schema_b, &aux );

	// Save the virtual microkernel address and the params.
	bli_auxinfo_set_ukr( gemm_ukr, &aux );
	bli_auxinfo_set_params( params, &aux );

	// We don't bother querying the thrinfo_t node for the 1st loop because
	// we can't parallelize that loop in trsm due to the inter-iteration
	// dependencies that exist.
	thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );
	//thrinfo_t* caucus = bli_thrinfo_sub_node( 0, thread );

	// Query the number of threads and thread ids for each loop.
	const dim_t jr_nt  = bli_thrinfo_n_way( thread );
	const dim_t jr_tid = bli_thrinfo_work_id( thread );

	dim_t jr_start, jr_end, jr_inc;

	// Determine the thread range and increment for the 2nd loop.
	// NOTE: The definition of bli_thread_range_slrr() will depend on whether
	// slab or round-robin partitioning was requested at configure-time.
	// NOTE: Parallelism in the 1st loop is unattainable due to the
	// inter-iteration dependencies present in trsm.
	bli_thread_range_slrr( jr_tid, jr_nt, n_iter, 1, FALSE, &jr_start, &jr_end, &jr_inc );

	// Loop over the n dimension (NR columns at a time).
	for ( dim_t j = jr_start; j < jr_end; j += jr_inc )
	{
		const char* b1 = b_cast + j * cstep_b;
		      char* c1 = c_cast + j * cstep_c;

		const dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left )
		                      ? NR : n_left );

		// Initialize our next panel of B to be the current panel of B.
		const char* b2 = b1;

		const char* a1  = a_cast;
		      char* c11 = c1 + (m_iter-1)*rstep_c;

		// Loop over the m dimension (MR rows at a time).
		for ( dim_t ib = 0; ib < m_iter; ++ib )
		{
			const dim_t  i          = m_iter - 1 - ib;
			const doff_t diagoffa_i = diagoffa + ( doff_t )i*MR;

			const dim_t m_cur = ( bli_is_not_edge_b( ib, m_iter, m_left )
			                      ? MR : m_left );

			// If the current panel of A intersects the diagonal, use a
			// special micro-kernel that performs a fused gemm and trsm.
			// If the current panel of A resides above the diagonal, use a
			// a regular gemm micro-kernel. Otherwise, if it is below the
			// diagonal, it was not packed (because it is implicitly zero)
			// and so we do nothing.
			if ( bli_intersects_diag_n( diagoffa_i, MR, k ) )
			{
				// Compute various offsets into and lengths of parts of A.
				const dim_t off_a11 = diagoffa_i;
				const dim_t k_a1112 = k - off_a11;
				const dim_t k_a11   = MR;
				const dim_t k_a12   = k_a1112 - MR;
				const dim_t off_a12 = off_a11 + k_a11;

				// Compute the panel stride for the current diagonal-
				// intersecting micro-panel.
				inc_t ps_a_cur  = k_a1112 * PACKMR;
				      ps_a_cur += ( bli_is_odd( ps_a_cur ) ? 1 : 0 );
				      ps_a_cur *= dt_a_size;

				// Compute the addresses of the triangular block A11 and the
				// panel A12.
				const char* a11 = a1;
				const char* a12 = a1 + k_a11 * PACKMR * dt_a_size;
				//a12 = bli_ptr_inc_by_frac( a1, sizeof( ctype ), k_a11 * PACKMR, 1 );

				// Compute the addresses of the panel B01 and the block
				// B11.
				const char* b11 = b1 + off_a11 * PACKNR * dt_b_size;
				const char* b21 = b1 + off_a12 * PACKNR * dt_b_size;

				// Compute the addresses of the next panels of A and B.
				const char* a2 = a1 + ps_a_cur;
				if ( bli_is_last_iter_rr( ib, m_iter, 0, 1 ) )
				{
					a2 = a_cast;
					b2 = b1;
					if ( bli_is_last_iter_slrr( j, n_iter, jr_tid, jr_nt ) )
						b2 = b_cast;
				}

				// Save addresses of next panels of A and B to the auxinfo_t
				// object.
				bli_auxinfo_set_next_a( a2, &aux );
				bli_auxinfo_set_next_b( b2, &aux );

				gemmtrsm_ukr
				(
				  m_cur,
				  n_cur,
				  k_a12,
				  ( void* )alpha1_cast,
				  ( void* )a12,
				  ( void* )a11,
				  ( void* )b21,
				  ( void* )b11,
				  c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);

				a1 += ps_a_cur;
			}
			else if ( bli_is_strictly_above_diag_n( diagoffa_i, MR, k ) )
			{
				// Compute the addresses of the next panels of A and B.
				const char* a2 = a1 + rstep_a;
				if ( bli_is_last_iter_rr( ib, m_iter, 0, 1 ) )
				{
					a2 = a_cast;
					b2 = b1;
					if ( bli_is_last_iter_slrr( j, n_iter, jr_tid, jr_nt ) )
						b2 = b_cast;
				}

				// Save addresses of next panels of A and B to the auxinfo_t
				// object.
				bli_auxinfo_set_next_a( a2, &aux );
				bli_auxinfo_set_next_b( b2, &aux );

				// Invoke the gemm micro-kernel.
				gemm_ukr
				(
				  m_cur,
				  n_cur,
				  k,
				  ( void* )minus_one,
				  ( void* )a1,
				  ( void* )b1,
				  ( void* )alpha2_cast,
				  c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);

				a1 += rstep_a;
			}

			c11 -= rstep_c;
		}
	}
}

/*
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: a1 (diag)", MR, k_a1112, a1, 1, MR, "%5.2f", "" );
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: b11 (diag)", MR, NR, b11, NR, 1, "%6.3f", "" );
printf( "m_iter     = %lu\n", m_iter );
printf( "m_cur      = %lu\n", m_cur );
printf( "k          = %lu\n", k );
printf( "diagoffa_i = %lu\n", diagoffa_i );
printf( "off_a1112  = %lu\n", off_a1112 );
printf( "k_a1112    = %lu\n", k_a1112 );
printf( "k_a12      = %lu\n", k_a12 );
printf( "k_a11      = %lu\n", k_a11 );
printf( "rs_c,cs_c  = %lu %lu\n", rs_c, cs_c );
printf( "rs_ct,cs_ct= %lu %lu\n", rs_ct, cs_ct );

PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: b11 after (diag)", MR, NR, b11, NR, 1, "%5.2f", "" );
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: b11 after (diag)", MR, NR, b11, NR, 1, "%5.2f", "" );
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: ct after (diag)", m_cur, n_cur, ct, rs_ct, cs_ct, "%5.2f", "" );
*/

