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

err_t bli_gemmsup_ref
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             rntm_t* rntm
     )
{
	// This function implements the default gemmsup handler. If you are a
	// BLIS developer and wish to use a different gemmsup handler, please
	// register a different function pointer in the context in your
	// sub-configuration's bli_cntx_init_*() function.

	// Don't use the small/unpacked implementation if one of the matrices
	// uses general stride. NOTE: We check for this here, in bli_gemmsup_ref()
	// (and not in the calling function, bli_gemmsup()), because we consider
	// this way of handling general stride to be part of the implementation
	// and not necessarily a general-purpose solution that would apply to all
	// possible gemmsup handlers. Similarly, we check for it here (and not in
	// the internal thread entry point, bli_gemmsup_int()) because we don't
	// want to have to manage the multiple return values from the threads,
	// which we would have to process into a single return value and then
	// return from the parallel/threaded region.
	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );
	if ( stor_id == BLIS_XXX ) return BLIS_FAILURE;

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop.
	bli_rntm_factorize_sup
	(
	  bli_obj_length( c ),
	  bli_obj_width( c ),
	  bli_obj_width( a ),
	  rntm
	);

#if 0
	printf( "rntm.pack_a = %d\n", ( int )bli_rntm_pack_a( rntm ) );
	printf( "rntm.pack_b = %d\n", ( int )bli_rntm_pack_b( rntm ) );

	//bli_rntm_set_pack_a( 0, rntm );
	//bli_rntm_set_pack_b( 0, rntm );
#endif

	return
	bli_l3_sup_thread_decorator
	(
	  bli_gemmsup_int,
	  BLIS_GEMM, // operation family id
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm
	);
}

// -----------------------------------------------------------------------------

err_t bli_gemmtsup_ref
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             rntm_t* rntm
     )
{
	// This function implements the default gemmtsup handler. If you are a
	// BLIS developer and wish to use a different gemmtsup handler, please
	// register a different function pointer in the context in your
	// sub-configuration's bli_cntx_init_*() function.

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_gemmt_check( alpha, a, b, beta, c, cntx );

#if 0
	// NOTE: This special case handling is done within the variants.

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// If A or B has a zero dimension, scale C by beta and return early.
	if ( bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return BLIS_SUCCESS;
	}
#endif

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop.
	bli_rntm_factorize_sup
	(
	  bli_obj_length( c ),
	  bli_obj_width( c ),
	  bli_obj_width( a ),
	  rntm
	);

	return
	bli_l3_sup_thread_decorator
	(
	  bli_gemmtsup_int,
	  BLIS_GEMMT, // operation family id
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm
	);
}

