/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2017 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of copyright holder(s) nor the names
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
#include "blix.h"

// Given the current architecture of BLIS sandboxes, bli_gemmnat() is the
// entry point to any sandbox implementation.

// NOTE: We must keep this function named bli_gemmnat() since this is the BLIS
// API function for which we are providing an alternative implementation via
// the sandbox.

void bli_gemmnat
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
    bli_init_once();

    // Obtain a valid (native) context from the gks if necessary.
    if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } 

    // Invoke the operation's front end.
    //blx_gemm_front( alpha, a, b, beta, c, cntx, rntm, NULL );
    blx_gemm_ref_var2( BLIS_NO_TRANSPOSE,
	                   alpha, a, b, beta, c,
	                   BLIS_XXX, cntx, rntm, NULL );
}

