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

void bli_trmm_ll_ker_var2
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

	// Detach and multiply the scalars attached to A and B.
	obj_t scalar_a, scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	const void* buf_alpha = bli_obj_internal_scalar_buffer( &scalar_b );
	const void* buf_beta  = bli_obj_internal_scalar_buffer( c );

	// Alias some constants to simpler names.
	const dim_t MR         = pd_a;
	const dim_t NR         = pd_b;
	const dim_t PACKMR     = cs_a;
	const dim_t PACKNR     = rs_b;

	// Query the context for the micro-kernel address and cast it to its
	// function pointer type.
	gemm_ukr_ft gemm_ukr   = bli_gemm_var_cntl_ukr( cntl );
	const void* params     = bli_gemm_var_cntl_params( cntl );

	const void* one        = bli_obj_buffer_for_const( dt_comp, &BLIS_ONE );
	const char* a_cast     = buf_a;
	const char* b_cast     = buf_b;
	      char* c_cast     = buf_c;
	const char* alpha_cast = buf_alpha;
	const char* beta_cast  = buf_beta;

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

	// Safeguard: If the current block of A is entirely above the diagonal,
	// it is implicitly zero. So we do nothing.
	if ( bli_is_strictly_above_diag_n( diagoffa, m, k ) ) return;

	// If there is a zero region above where the diagonal of A intersects the
	// left edge of the block, adjust the pointer to C and treat this case as
	// if the diagonal offset were zero. This skips over the region that was
	// not packed. (Note we assume the diagonal offset is a multiple of MR;
	// this assumption will hold as long as the cache blocksizes are each a
	// multiple of MR and NR.)
	if ( diagoffa < 0 )
	{
		m        += diagoffa;
		c_cast   -= diagoffa * rs_c * dt_c_size;
		diagoffa  = 0;
	}

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

	// The 'thread' argument points to the thrinfo_t node for the 2nd (jr)
	// loop around the microkernel. Here we query the thrinfo_t node for the
	// 1st (ir) loop around the microkernel.
	thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );
	//thrinfo_t* caucus = bli_thrinfo_sub_node( 0, thread );

	// Query the number of threads and thread ids for each loop.
	const dim_t jr_nt  = bli_thrinfo_n_way( thread );
	const dim_t jr_tid = bli_thrinfo_work_id( thread );
	//const dim_t ir_nt  = bli_thrinfo_n_way( caucus );
	//const dim_t ir_tid = bli_thrinfo_work_id( caucus );

	dim_t jr_start, jr_end, jr_inc;

	// Determine the thread range and increment for the 2nd loop.
	// NOTE: The definition of bli_thread_range_slrr() will depend on whether
	// slab or round-robin partitioning was requested at configure-time.
	// NOTE: Parallelism in the 1st loop is disabled for now.
	bli_thread_range_slrr( jr_tid, jr_nt, n_iter, 1, FALSE, &jr_start, &jr_end, &jr_inc );
	//bli_thread_range_rr( ir_tid, ir_nt, m_iter, 1, FALSE, &ir_start, &ir_end, &ir_inc );

	// Loop over the n dimension (NR columns at a time).
	for ( dim_t j = jr_start; j < jr_end; j += jr_inc )
	{
		const char* b1 = b_cast + j * cstep_b;
		      char* c1 = c_cast + j * cstep_c;

		const dim_t n_cur = ( bli_is_not_edge_f( j, n_iter, n_left )
		                      ? NR : n_left );

		// Initialize our next panel of B to be the current panel of B.
		const char* b2 = b1;

		// Initialize pointers for stepping through the block of A and current
		// column of microtiles of C.
		const char* a1  = a_cast;
		      char* c11 = c1;

		// Loop over the m dimension (MR rows at a time).
		for ( dim_t i = 0; i < m_iter; ++i )
		{
			const doff_t diagoffa_i = diagoffa + ( doff_t )i*MR;

			const dim_t m_cur = ( bli_is_not_edge_f( i, m_iter, m_left )
			                      ? MR : m_left );

			// If the current panel of A intersects the diagonal, scale C
			// by beta. If it is strictly below the diagonal, scale by one.
			// This allows the current macro-kernel to work for both trmm
			// and trmm3.
			if ( bli_intersects_diag_n( diagoffa_i, MR, k ) )
			{
				// Determine the offset to and length of the panel that was
				// packed so we can index into the corresponding location in
				// b1.
				const dim_t off_a1011 = 0;
				const dim_t k_a1011   = bli_min( diagoffa_i + MR, k );

				// Compute the panel stride for the current diagonal-
				// intersecting micro-panel.
				inc_t ps_a_cur  = k_a1011 * PACKMR;
				      ps_a_cur += ( bli_is_odd( ps_a_cur ) ? 1 : 0 );
				      ps_a_cur *= dt_a_size;

				// NOTE: ir loop parallelism disabled for now.
				//if ( bli_trmm_my_iter( i, ir_thread ) ) {

				const char* b1_i = b1 + off_a1011 * PACKNR * dt_b_size;

				// Compute the addresses of the next panels of A and B.
				const char* a2 = bli_trmm_get_next_a_upanel( a1, rstep_a, 1 );
				if ( bli_is_last_iter_slrr( i, m_iter, 0, 1 ) )
				{
					a2 = a_cast;
					b2 = bli_trmm_get_next_b_upanel( b1, cstep_b, jr_inc );
					//if ( bli_is_last_iter_slrr( j, n_iter, jr_tid, jr_nt ) )
					//	b2 = b_cast;
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
				  k_a1011,
				  ( void* )alpha_cast,
				  ( void* )a1,
				  ( void* )b1_i,
				  ( void* )beta_cast,
				  c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);
				//}

				a1 += ps_a_cur;
			}
			else if ( bli_is_strictly_below_diag_n( diagoffa_i, MR, k ) )
			{
				// NOTE: ir loop parallelism disabled for now.
				//if ( bli_trmm_my_iter( i, ir_thread ) ) {

				// Compute the addresses of the next panels of A and B.
				const char* a2 = bli_trmm_get_next_a_upanel( a1, rstep_a, 1 );
				if ( bli_is_last_iter_slrr( i, m_iter, 0, 1 ) )
				{
					a2 = a_cast;
					b2 = bli_trmm_get_next_b_upanel( b1, cstep_b, jr_inc );
					//if ( bli_is_last_iter_slrr( j, n_iter, jr_tid, jr_nt ) )
					//	b2 = b_cast;
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
				  ( void* )alpha_cast,
				  ( void* )a1,
				  ( void* )b1,
				  ( void* )one,
				  c11, rs_c, cs_c,
				  &aux,
				  ( cntx_t* )cntx
				);
				//}

				a1 += rstep_a;
			}

			c11 += rstep_c;
		}
	}
}

//PASTEMAC(ch,printm)( "trmm_ll_ker_var2: a1", MR, k_a1011, a1,   1, MR, "%4.1f", "" );
//PASTEMAC(ch,printm)( "trmm_ll_ker_var2: b1", k_a1011, NR, b1_i, NR, 1, "%4.1f", "" );

