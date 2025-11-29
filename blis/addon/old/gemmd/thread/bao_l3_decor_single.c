/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

#define SKIP_THRINFO_TREE

void bao_l3_thread_decorator_single
     (
       l3aoint_ft func,
       opid_t     family,
       //pack_t     schema_a,
       //pack_t     schema_b,
       obj_t*     alpha,
       obj_t*     a,
       obj_t*     d,
       obj_t*     b,
       obj_t*     beta,
       obj_t*     c,
       cntx_t*    cntx,
       rntm_t*    rntm
     )
{
	// For sequential execution, we use only one thread.
	const dim_t n_threads = 1;

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* restrict array = bli_sba_checkout_array( n_threads );

	// Access the pool_t* for thread 0 and embed it into the rntm.
	bli_sba_rntm_set_pool( 0, array, rntm );

	// Set the packing block allocator field of the rntm.
	bli_pba_rntm_set_pba( rntm );

#ifndef SKIP_THRINFO_TREE
	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t* restrict gl_comm = bli_thrcomm_create( rntm, n_threads );
#endif


	{
		// NOTE: We don't need to create another copy of the rntm_t since
		// it was already copied in one of the high-level oapi functions.
		rntm_t* restrict rntm_p = rntm;

		// There is only one thread id (for the thief thread).
		const dim_t tid = 0;

		// Use the thread id to access the appropriate pool_t* within the
		// array_t, and use it to set the sba_pool field within the rntm_t.
		// If the pool_t* element within the array_t is NULL, it will first
		// be allocated/initialized.
		// NOTE: This is commented out because, in the single-threaded case,
		// this is redundant since it's already been done above.
		//bli_sba_rntm_set_pool( tid, array, rntm_p );

#ifndef SKIP_THRINFO_TREE
		thrinfo_t* thread = NULL;

		// Create the root node of the thread's thrinfo_t structure.
		bli_l3_sup_thrinfo_create_root( tid, gl_comm, rntm_p, &thread );
#else
		// This optimization allows us to use one of the global thrinfo_t
		// objects for single-threaded execution rather than grow one from
		// scratch. The key is that bli_thrinfo_sup_grow(), which is called
		// from within the variants, will immediately return if it detects
		// that the thrinfo_t* passed into it is either
		// &BLIS_GEMM_SINGLE_THREADED or &BLIS_PACKM_SINGLE_THREADED.
		thrinfo_t* thread = &BLIS_GEMM_SINGLE_THREADED;

		( void )tid;
#endif

		func
		(
		  alpha,
		  a,
		  d,
		  b,
		  beta,
		  c,
		  cntx,
		  rntm_p,
		  thread
		);

#ifndef SKIP_THRINFO_TREE
		// Free the current thread's thrinfo_t structure.
		bli_l3_sup_thrinfo_free( rntm_p, thread );
#endif
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called above).

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );
}

