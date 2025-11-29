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

void bli_unpackm_int
     (
       const obj_t*     a,
             obj_t*     p,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	bli_init_once();

	unpackm_var_oft f;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_unpackm_int_check( p, a, cntx );

	// If p was aliased to a during the pack stage (because it was already
	// in an acceptable packed/contiguous format), then no unpack is actually
	// necessary, so we return.
	if ( bli_obj_is_alias_of( p, a ) ) return;

	// Barrier so that we know threads are done with previous computation
	// with the same packing buffer before starting to pack.
	thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );
	bli_thrinfo_barrier( thread );

	// Extract the function pointer from the current control tree node.
	f = bli_cntl_unpackm_params_var_func( cntl );

	// Invoke the variant.
	f
	(
	  p,
	  a,
	  cntx,
	  cntl,
	  thread
	);

	// Barrier so that unpacking is done before computation.
	bli_thrinfo_barrier( thread );
}

