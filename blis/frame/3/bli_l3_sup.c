/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

err_t bli_gemmsup
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	// Return early if small matrix handling is disabled at configure-time.
	#ifdef BLIS_DISABLE_SUP_HANDLING
	return BLIS_FAILURE;
	#endif

	// Return early if this is a mixed-datatype computation.
	if ( bli_obj_dt( c ) != bli_obj_dt( a ) ||
	     bli_obj_dt( c ) != bli_obj_dt( b ) ||
	     bli_obj_comp_prec( c ) != bli_obj_prec( c ) ) return BLIS_FAILURE;

	// Obtain a valid (native) context from the gks if necessary.
	// NOTE: This must be done before calling the _check() function, since
	// that function assumes the context pointer is valid.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	const num_t dt = bli_obj_dt( c );
	const dim_t m  = bli_obj_length( c );
	const dim_t n  = bli_obj_width( c );
	const dim_t k  = bli_obj_width_after_trans( a );

	// Return early if a microkernel preference-induced transposition would
	// have been performed and shifted the dimensions outside of the space
	// of sup-handled problems.
	if ( bli_cntx_dislikes_storage_of( c, BLIS_GEMM_UKR, cntx ) )
	{
		// Pass in m and n reversed, which simulates a transposition of the
		// entire operation pursuant to the microkernel storage preference.
		if ( !bli_cntx_l3_sup_thresh_is_met( dt, n, m, k, cntx ) )
			return BLIS_FAILURE;
	}
	else // ukr_prefers_storage_of( c, ... )
	{
		if ( !bli_cntx_l3_sup_thresh_is_met( dt, m, n, k, cntx ) )
			return BLIS_FAILURE;
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	if ( !bli_rntm_l3_sup( &rntm_l ) )
		return BLIS_FAILURE;

#if 0
const num_t dt = bli_obj_dt( c );
const dim_t m  = bli_obj_length( c );
const dim_t n  = bli_obj_width( c );
const dim_t k  = bli_obj_width_after_trans( a );
const dim_t tm = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_MT, cntx );
const dim_t tn = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_NT, cntx );
const dim_t tk = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_KT, cntx );

printf( "dims: %d %d %d (threshs: %d %d %d)\n",
        (int)m, (int)n, (int)k, (int)tm, (int)tn, (int)tk );
#endif

	// We've now ruled out the following two possibilities:
	// - the ukernel prefers the operation as-is, and the sup thresholds are
	//   unsatisfied.
	// - the ukernel prefers a transposed operation, and the sup thresholds are
	//   unsatisfied after taking into account the transposition.
	// This implies that the sup thresholds (at least one of them) are met.
	// and the small/unpacked handler should be called.
	// NOTE: The sup handler is free to enforce a stricter threshold regime
	// if it so chooses, in which case it can/should return BLIS_FAILURE.

	// Query the small/unpacked handler from the context and invoke it.
	gemmsup_oft gemmsup_fp = bli_cntx_get_l3_sup_handler( BLIS_GEMM, cntx );

	// Typecast alpha and beta to the correct type
	obj_t alpha_cast, beta_cast;
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, alpha, &alpha_cast );
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, beta, &beta_cast );

	return
	gemmsup_fp
	(
	  &alpha_cast,
	  a,
	  b,
	  &beta_cast,
	  c,
	  cntx,
	  &rntm_l
	);
}


err_t bli_gemmtsup
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	// Return early if small matrix handling is disabled at configure-time.
	#ifdef BLIS_DISABLE_SUP_HANDLING
	return BLIS_FAILURE;
	#endif

	// Return early if this is a mixed-datatype computation.
	if ( bli_obj_dt( c ) != bli_obj_dt( a ) ||
	     bli_obj_dt( c ) != bli_obj_dt( b ) ||
	     bli_obj_comp_prec( c ) != bli_obj_prec( c ) ) return BLIS_FAILURE;

	// Obtain a valid (native) context from the gks if necessary.
	// NOTE: This must be done before calling the _check() function, since
	// that function assumes the context pointer is valid.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Return early if the problem dimensions exceed their sup thresholds.
	// Notice that we do not bother to check whether the microkernel
	// prefers or dislikes the storage of C, since the same check is called
	// for either way.
	const num_t dt = bli_obj_dt( c );
	const dim_t m  = bli_obj_length( c );
	const dim_t k  = bli_obj_width_after_trans( a );

	if ( !bli_cntx_l3_sup_thresh_is_met( dt, m, m, k, cntx ) )
		return BLIS_FAILURE;

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// We've now ruled out the possibility that the sup thresholds are
	// unsatisfied.
	// This implies that the sup thresholds (at least one of them) are met.
	// and the small/unpacked handler should be called.
	// NOTE: The sup handler is free to enforce a stricter threshold regime
	// if it so chooses, in which case it can/should return BLIS_FAILURE.

	// Query the small/unpacked handler from the context and invoke it.
	gemmtsup_oft gemmtsup_fp = bli_cntx_get_l3_sup_handler( BLIS_GEMMT, cntx );

	// Typecast alpha and beta to the correct type
	obj_t alpha_cast, beta_cast;
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, alpha, &alpha_cast );
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, beta, &beta_cast );

	return
	gemmtsup_fp
	(
	  &alpha_cast,
	  a,
	  b,
	  &beta_cast,
	  c,
	  cntx,
	  &rntm_l
	);
}


