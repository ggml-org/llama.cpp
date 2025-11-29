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

void bli_trsm_rl_ker_var2
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

	      doff_t diagoffb  = bli_obj_diag_offset( b );

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
	// attached to A (the non-triangular matrix). This will be the alpha
	// scalar used in the gemmtrsm subproblems (ie: the scalar that would
	// be applied to the packed copy of A prior to it being updated by
	// the trsm subproblem). This scalar may be unit, if for example it
	// was applied during packing.
	const void* buf_alpha1 = bli_obj_internal_scalar_buffer( a );

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
	// NOTE: We use the upper-triangular gemmtrsm ukernel because, while
	// the current macro-kernel targets the "rl" case (right-side/lower-
	// triangular), it becomes upper-triangular after the kernel operation
	// is transposed so that all kernel instances are of the "left"
	// variety (since those are the only trsm ukernels that exist).
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
	     cs_a == PACKNR
	     pd_a == NR
	     ps_a == stride to next micro-panel of A
	     rs_b == PACKMR
	     cs_b == 1
	     pd_b == MR
	     ps_b == stride to next micro-panel of B
	     rs_c == (no assumptions)
	     cs_c == (no assumptions)

	  Note that MR/NR and PACKMR/PACKNR have been swapped to reflect the
	  swapping of values in the control tree (ie: those values used when
	  packing). This swapping is needed since we cast right-hand trsm in
	  terms of transposed left-hand trsm. So, if we're going to be
	  transposing the operation, then A needs to be packed with NR and B
	  needs to be packed with MR (remember: B is the triangular matrix in
	  the right-hand side parameter case).
	*/

	// Safety trap: Certain indexing within this macro-kernel does not
	// work as intended if both MR and NR are odd.
	if ( ( bli_is_odd( PACKMR ) && bli_is_odd( NR ) ) ||
	     ( bli_is_odd( PACKNR ) && bli_is_odd( MR ) ) ) bli_abort();

	// If any dimension is zero, return immediately.
	if ( bli_zero_dim3( m, n, k ) ) return;

	// Safeguard: If the current panel of B is entirely above its diagonal,
	// it is implicitly zero. So we do nothing.
	if ( bli_is_strictly_above_diag_n( diagoffb, k, n ) ) return;

	// If there is a zero region above where the diagonal of B intersects
	// the left edge of the panel, adjust the pointer to A and treat this
	// case as if the diagonal offset were zero. Note that we don't need to
	// adjust the pointer to B since packm would have simply skipped over
	// the region that was not stored.
	if ( diagoffb < 0 )
	{
		k        += diagoffb;
		a_cast   -= diagoffb * PACKMR * dt_a_size;
		diagoffb  = 0;
	}

	// If there is a zero region to the right of where the diagonal
	// of B intersects the bottom of the panel, shrink it so that
	// we can index to the correct place in C (corresponding to the
	// part of the panel of B that was packed).
	// NOTE: This is NOT being done to skip over "no-op" iterations,
	// as with the trsm_lu macro-kernel. This MUST be done for correct
	// execution because we use n (via n_iter) to compute diagonal and
	// index offsets for backwards movement through B.
	if ( diagoffb + k < n )
	{
		n = diagoffb + k;
	}

	// Check the k dimension, which needs to be a multiple of NR. If k
	// isn't a multiple of NR, we adjust it higher to satisfy the micro-
	// kernel, which is expecting to perform an NR x NR triangular solve.
	// This adjustment of k is consistent with what happened when B was
	// packed: all of its bottom/right edges were zero-padded, and
	// furthermore, the panel that stores the bottom-right corner of the
	// matrix has its diagonal extended into the zero-padded region (as
	// identity). This allows the trsm of that bottom-right panel to
	// proceed without producing any infs or NaNs that would infect the
	// "good" values of the corresponding block of A.
	if ( k % NR != 0 ) k += NR - ( k % NR );

	// NOTE: We don't need to check that n is a multiple of PACKNR since we
	// know that the underlying buffer was already allocated to have an n
	// dimension that is a multiple of PACKNR, with the region between the
	// last column and the next multiple of NR zero-padded accordingly.

	thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );

	// Compute number of primary and leftover components of the m and n
	// dimensions.
	dim_t n_iter = n / NR;
	dim_t n_left = n % NR;

	dim_t m_iter = m / MR;
	dim_t m_left = m % MR;

	if ( n_left ) ++n_iter;
	if ( m_left ) ++m_iter;

	// Determine some increments used to step through A, B, and C.
	inc_t rstep_a = ps_a * dt_a_size;

	inc_t cstep_b = ps_b * dt_b_size;

	inc_t rstep_c = rs_c * MR * dt_c_size;
	inc_t cstep_c = cs_c * NR * dt_c_size;

	auxinfo_t aux;

	// Save the pack schemas of A and B to the auxinfo_t object.
	// NOTE: We swap the values for A and B since the triangular
	// "A" matrix is actually contained within B.
	bli_auxinfo_set_schema_a( schema_b, &aux );
	bli_auxinfo_set_schema_b( schema_a, &aux );

	// Save the virtual microkernel address and the params.
	bli_auxinfo_set_ukr( gemm_ukr, &aux );
	bli_auxinfo_set_params( params, &aux );

	const char* b1 = b_cast;
	      char* c1 = c_cast;

	// Loop over the n dimension (NR columns at a time).
	for ( dim_t jb = 0; jb < n_iter; ++jb )
	{
		dim_t  j          = n_iter - 1 - jb;
		doff_t diagoffb_j = diagoffb - ( doff_t )j*NR;

		dim_t  n_cur = ( bli_is_not_edge_b( jb, n_iter, n_left ) ? NR : n_left );

		const char* a1         = a_cast;
		      char* c11        = c1 + (n_iter-1)*cstep_c;

		// Initialize our next panel of B to be the current panel of B.
		const char* b2 = b1;

		// If the current panel of B intersects the diagonal, use a
		// special micro-kernel that performs a fused gemm and trsm.
		// If the current panel of B resides below the diagonal, use a
		// a regular gemm micro-kernel. Otherwise, if it is above the
		// diagonal, it was not packed (because it is implicitly zero)
		// and so we do nothing.
		if ( bli_intersects_diag_n( diagoffb_j, k, NR ) )
		{
			// Determine the offset to and length of the panel that was packed
			// so we can index into the corresponding location in A.
			dim_t off_b11   = bli_max( -diagoffb_j, 0 );
			dim_t k_b1121   = k - off_b11;
			dim_t k_b11     = NR;
			dim_t k_b21     = k_b1121 - NR;
			dim_t off_b21   = off_b11 + k_b11;

			// Compute the addresses of the triangular block B11 and the
			// panel B21.
			const char* b11 = b1;
			const char* b21 = b1 + k_b11 * PACKNR * dt_b_size;
			//b21 = bli_ptr_inc_by_frac( b1, sizeof( ctype ), k_b11 * PACKNR, 1 );

			// Compute the panel stride for the current micro-panel.
			inc_t ps_b_cur  = k_b1121 * PACKNR;
				  ps_b_cur += ( bli_is_odd( ps_b_cur ) ? 1 : 0 );
				  ps_b_cur *= dt_b_size;

			// Loop over the m dimension (MR rows at a time).
			for ( dim_t i = 0; i < m_iter; ++i )
			{
				if ( bli_trsm_my_iter_rr( i, thread ) ){

				dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left );

				// Compute the addresses of the A11 block and A12 panel.
				const char* a11  = a1 + off_b11 * PACKMR * dt_a_size;
				const char* a12  = a1 + off_b21 * PACKMR * dt_a_size;

				// Compute the addresses of the next panels of A and B.
				const char* a2 = a1;
				//if ( bli_is_last_iter_rr( i, m_iter, 0, 1 ) )
				if ( i + bli_thrinfo_num_threads(thread) >= m_iter )
				{
					a2 = a_cast;
					b2 = b1 + ps_b_cur;
					if ( bli_is_last_iter_rr( jb, n_iter, 0, 1 ) )
						b2 = b_cast;
				}

				// Save addresses of next panels of A and B to the auxinfo_t
				// object. NOTE: We swap the values for A and B since the
				// triangular "A" matrix is actually contained within B.
				bli_auxinfo_set_next_a( b2, &aux );
				bli_auxinfo_set_next_b( a2, &aux );

				gemmtrsm_ukr
				(
				  m_cur,
				  n_cur,
				  k_b21,
				  ( void* )alpha1_cast,
				  ( void* )b21,
				  ( void* )b11,
				  ( void* )a12,
				  ( void* )a11,
				  c11, cs_c, rs_c,
				  &aux,
				  ( cntx_t* )cntx
				);

				}

				a1  += rstep_a;
				c11 += rstep_c;
			}

			b1 += ps_b_cur;
		}
		else if ( bli_is_strictly_below_diag_n( diagoffb_j, k, NR ) )
		{
			// Loop over the m dimension (MR rows at a time).
			for ( dim_t i = 0; i < m_iter; ++i )
			{
				if ( bli_trsm_my_iter_rr( i, thread ) ){

				dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left );

				// Compute the addresses of the next panels of A and B.
				const char* a2 = a1;
				//if ( bli_is_last_iter_rr( i, m_iter, 0, 1 ) )
				if ( i + bli_thrinfo_num_threads(thread) >= m_iter )
				{
					a2 = a_cast;
					b2 = b1 + cstep_b;
					if ( bli_is_last_iter_rr( jb, n_iter, 0, 1 ) )
						b2 = b_cast;
				}

				// Save addresses of next panels of A and B to the auxinfo_t
				// object. NOTE: We swap the values for A and B since the
				// triangular "A" matrix is actually contained within B.
				bli_auxinfo_set_next_a( b2, &aux );
				bli_auxinfo_set_next_b( a2, &aux );

				// Invoke the gemm micro-kernel.
				gemm_ukr
				(
				  m_cur,
				  n_cur,
				  k,
				  ( void* )minus_one,
				  ( void* )b1,
				  ( void* )a1,
				  ( void* )alpha2_cast,
				  c11, cs_c, rs_c,
				  &aux,
				  ( cntx_t* )cntx
				);

				}

				a1  += rstep_a;
				c11 += rstep_c;
			}

			b1 += cstep_b;
		}

		c1 -= cstep_c;
	}
}

