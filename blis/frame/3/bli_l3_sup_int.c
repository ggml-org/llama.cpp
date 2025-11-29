/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2020, Advanced Micro Devices, Inc.

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

err_t bli_gemmsup_int
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm,
             thrinfo_t* thread
     )
{
#if 0
	//bli_gemmsup_ref_var2
	//bli_gemmsup_ref_var1
	#if 0
	bli_gemmsup_ref_var1n
	#else
	#endif
	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );
	const bool    is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	if ( is_rrr_rrc_rcr_crr )
	{
		bli_gemmsup_ref_var2m
		(
		  BLIS_NO_TRANSPOSE, alpha, a, b, beta, c, stor_id, cntx, rntm
		);
	}
	else
	{
		bli_gemmsup_ref_var2m
		(
		  BLIS_TRANSPOSE, alpha, a, b, beta, c, stor_id, cntx, rntm
		);
	}

	return BLIS_SUCCESS;
#endif

	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

	const bool    is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	const bool    is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;

	const num_t   dt         = bli_obj_dt( c );
	const bool    row_pref   = bli_cntx_ukr_prefers_rows_dt( dt, bli_stor3_ukr( stor_id ), cntx );

	const bool    is_primary = ( row_pref ? is_rrr_rrc_rcr_crr
	                                      : is_rcc_crc_ccr_ccc );

	const dim_t  m           = bli_obj_length( c );
	const dim_t  n           = bli_obj_width( c );
	const dim_t  MR          = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t  NR          = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	const bool   auto_factor = bli_rntm_auto_factor( rntm );
	const dim_t  n_threads   = bli_rntm_num_threads( rntm );
	bool         use_bp      = TRUE;
	dim_t        jc_new;
	dim_t        ic_new;


	if ( is_primary )
	{
		// This branch handles:
		//  - rrr rrc rcr crr for row-preferential kernels
		//  - rcc crc ccr ccc for column-preferential kernels

		const dim_t mu = m / MR;
		const dim_t nu = n / NR;

		// Decide which algorithm to use (block-panel var2m or panel-block
		// var1n) based on the number of micropanels in the m and n dimensions.
		// Also, recalculate the automatic thread factorization.
		if         ( mu >= nu )    use_bp = TRUE;
		else /* if ( mu <  nu ) */ use_bp = FALSE;

		// If the parallel thread factorization was automatic, we update it
		// with a new factorization based on the matrix dimensions in units
		// of micropanels.
		if ( auto_factor )
		{
			if ( use_bp )
			{
				// In the block-panel algorithm, the m dimension is parallelized
				// with ic_nt and the n dimension is parallelized with jc_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );
			}
			else // if ( !use_bp )
			{
				// In the panel-block algorithm, the m dimension is parallelized
				// with jc_nt and the n dimension is parallelized with ic_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &jc_new, &ic_new );
			}

			// Update the ways of parallelism for the jc and ic loops, and then
			// update the current thread's root thrinfo_t node according to the
			// new ways of parallelism value for the jc loop.
			rntm_t rntm_l = *rntm;
			bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, &rntm_l );
			bli_l3_sup_thrinfo_update( &rntm_l, &thread );
		}


		if ( use_bp )
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var2m primary\n" );
			#endif
			// block-panel macrokernel; m -> mc, mr; n -> nc, nr: var2()
			bli_gemmsup_ref_var2m( BLIS_NO_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, thread );
		}
		else // use_pb
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var1n primary\n" );
			#endif
			// panel-block macrokernel; m -> nc*,mr; n -> mc*,nr: var1()
			bli_gemmsup_ref_var1n( BLIS_NO_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, thread );
			// *requires nudging of nc up to be a multiple of mr.
		}
	}
	else
	{
		// This branch handles:
		//  - rrr rrc rcr crr for column-preferential kernels
		//  - rcc crc ccr ccc for row-preferential kernels

		const dim_t mu = n / MR; // the n becomes m after a transposition
		const dim_t nu = m / NR; // the m becomes n after a transposition

		// Decide which algorithm to use (block-panel var2m or panel-block
		// var1n) based on the number of micropanels in the m and n dimensions.
		// Also, recalculate the automatic thread factorization.
		if         ( mu >= nu )    use_bp = TRUE;
		else /* if ( mu <  nu ) */ use_bp = FALSE;

		// If the parallel thread factorization was automatic, we update it
		// with a new factorization based on the matrix dimensions in units
		// of micropanels.
		if ( auto_factor )
		{
			if ( use_bp )
			{
				// In the block-panel algorithm, the m dimension is parallelized
				// with ic_nt and the n dimension is parallelized with jc_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );
			}
			else // if ( !use_bp )
			{
				// In the panel-block algorithm, the m dimension is parallelized
				// with jc_nt and the n dimension is parallelized with ic_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &jc_new, &ic_new );
			}

			// Update the ways of parallelism for the jc and ic loops, and then
			// update the current thread's root thrinfo_t node according to the
			// new ways of parallelism value for the jc loop.
			rntm_t rntm_l = *rntm;
			bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, &rntm_l );
			bli_l3_sup_thrinfo_update( &rntm_l, &thread );
		}


		if ( use_bp )
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var2m non-primary\n" );
			#endif
			// panel-block macrokernel; m -> nc, nr; n -> mc, mr: var2() + trans
			bli_gemmsup_ref_var2m( BLIS_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, thread );
		}
		else // use_pb
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var1n non-primary\n" );
			#endif
			// block-panel macrokernel; m -> mc*,nr; n -> nc*,mr: var1() + trans
			bli_gemmsup_ref_var1n( BLIS_TRANSPOSE,
			                       alpha, a, b, beta, c,
			                       stor_id, cntx, rntm, thread );
			// *requires nudging of mc up to be a multiple of nr.
		}
	}

	// Return success so that the caller knows that we computed the solution.
	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

err_t bli_gemmtsup_int
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm,
             thrinfo_t* thread
     )
{
	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

	const bool    is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	const bool    is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;

	const num_t   dt         = bli_obj_dt( c );
	const bool    row_pref   = bli_cntx_ukr_prefers_rows_dt( dt, bli_stor3_ukr( stor_id ), cntx );

	const bool    is_primary = ( row_pref ? is_rrr_rrc_rcr_crr
	                                      : is_rcc_crc_ccr_ccc );

	const dim_t  m           = bli_obj_length( c );
	const dim_t  n           = m;
	const dim_t  MR          = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t  NR          = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	const bool   auto_factor = bli_rntm_auto_factor( rntm );
	const dim_t  n_threads   = bli_rntm_num_threads( rntm );
	bool         use_bp      = TRUE;
	dim_t        jc_new;
	dim_t        ic_new;


	if ( is_primary )
	{
		// This branch handles:
		//  - rrr rrc rcr crr for row-preferential kernels
		//  - rcc crc ccr ccc for column-preferential kernels

		const dim_t mu = m / MR;
		const dim_t nu = n / NR;

		// Decide which algorithm to use (block-panel var2m or panel-block
		// var1n) based on the number of micropanels in the m and n dimensions.
		// Also, recalculate the automatic thread factorization.
		if         ( mu >= nu )    use_bp = TRUE;
		else /* if ( mu <  nu ) */ use_bp = FALSE;

		// If the parallel thread factorization was automatic, we update it
		// with a new factorization based on the matrix dimensions in units
		// of micropanels.
		if ( auto_factor )
		{
			if ( use_bp )
			{
				// In the block-panel algorithm, the m dimension is parallelized
				// with ic_nt and the n dimension is parallelized with jc_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );
			}
			else // if ( !use_bp )
			{
				// In the panel-block algorithm, the m dimension is parallelized
				// with jc_nt and the n dimension is parallelized with ic_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &jc_new, &ic_new );
			}

			// Update the ways of parallelism for the jc and ic loops, and then
			// update the current thread's root thrinfo_t node according to the
			// new ways of parallelism value for the jc loop.
			rntm_t rntm_l = *rntm;
			bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, &rntm_l );
			bli_l3_sup_thrinfo_update( &rntm_l, &thread );
		}


		if ( use_bp )
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var2m primary\n" );
			#endif
			// block-panel macrokernel; m -> mc, mr; n -> nc, nr: var2()
#if 0
			bli_gemmtsup_ref_var2m( BLIS_NO_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
#endif
		}
		else // use_pb
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var1n primary\n" );
			#endif
			// panel-block macrokernel; m -> nc*,mr; n -> mc*,nr: var1()
#if 0
			bli_gemmtsup_ref_var1n( BLIS_NO_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
#endif
			// *requires nudging of nc up to be a multiple of mr.
		}
	}
	else
	{
		// This branch handles:
		//  - rrr rrc rcr crr for column-preferential kernels
		//  - rcc crc ccr ccc for row-preferential kernels

		const dim_t mu = n / MR; // the n becomes m after a transposition
		const dim_t nu = m / NR; // the m becomes n after a transposition

		// Decide which algorithm to use (block-panel var2m or panel-block
		// var1n) based on the number of micropanels in the m and n dimensions.
		// Also, recalculate the automatic thread factorization.
		if         ( mu >= nu )    use_bp = TRUE;
		else /* if ( mu <  nu ) */ use_bp = FALSE;

		// If the parallel thread factorization was automatic, we update it
		// with a new factorization based on the matrix dimensions in units
		// of micropanels.
		if ( auto_factor )
		{
			if ( use_bp )
			{
				// In the block-panel algorithm, the m dimension is parallelized
				// with ic_nt and the n dimension is parallelized with jc_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );
			}
			else // if ( !use_bp )
			{
				// In the panel-block algorithm, the m dimension is parallelized
				// with jc_nt and the n dimension is parallelized with ic_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &jc_new, &ic_new );
			}

			// Update the ways of parallelism for the jc and ic loops, and then
			// update the current thread's root thrinfo_t node according to the
			// new ways of parallelism value for the jc loop.
			rntm_t rntm_l = *rntm;
			bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, &rntm_l );
			bli_l3_sup_thrinfo_update( &rntm_l, &thread );
		}


		if ( use_bp )
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var2m non-primary\n" );
			#endif
			// panel-block macrokernel; m -> nc, nr; n -> mc, mr: var2() + trans
#if 0
			bli_gemmtsup_ref_var2m( BLIS_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
#endif
		}
		else // use_pb
		{
			#ifdef TRACEVAR
			if ( bli_thrinfo_am_chief( thread ) )
			printf( "bli_l3_sup_int(): var1n non-primary\n" );
			#endif
			// block-panel macrokernel; m -> mc*,nr; n -> nc*,mr: var1() + trans
#if 0
			bli_gemmtsup_ref_var1n( BLIS_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
#endif
			// *requires nudging of mc up to be a multiple of nr.
		}
	}

	// Return success so that the caller knows that we computed the solution.
	return BLIS_SUCCESS;
}

