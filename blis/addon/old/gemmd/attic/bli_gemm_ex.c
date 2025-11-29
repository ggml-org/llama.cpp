/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

void bli_gemm_ex
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
	bli_init_once();

	// A switch to easily toggle whether we use the addon implementation
	// of bao_gemmd() as the implementation for bli_gemm(). (This allows for
	// easy testing of bao_gemmd() via the testsuite.)
	if ( 1 )
	{
		const dim_t k  = bli_obj_width_after_trans( a );
		const num_t dt = bli_obj_dt( c );
		obj_t       d;

		bli_obj_create( dt, k, 1, 1, k, &d );
		bli_setv( &BLIS_ONE, &d );
		//bli_randv( &d );

		bao_gemmd_ex( alpha, a, &d, b, beta, c, cntx, rntm );

		bli_obj_free( &d );
		return;
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
	else                { rntm_l = *rntm;                       rntm = &rntm_l; }

	// Obtain a valid (native) context from the gks if necessary.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

	// Invoke the operation's front end.
	bli_gemm_front
	(
	  ( obj_t* )alpha, ( obj_t* )a, ( obj_t* )b, ( obj_t* )beta, ( obj_t* )c,
	  ( cntx_t* )cntx, ( rntm_t* )rntm, NULL
	);
}

