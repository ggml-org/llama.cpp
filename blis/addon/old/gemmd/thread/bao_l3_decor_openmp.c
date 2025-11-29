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

#ifdef BLIS_ENABLE_OPENMP

//#define PRINT_THRINFO

void bao_l3_thread_decorator_openmp
     (
       l3aoint_ft func,
       opid_t     family,
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
	// Query the total number of threads from the rntm_t object.
	const dim_t n_threads = bli_rntm_num_threads( rntm );

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* array = bli_sba_checkout_array( n_threads );

	// Access the pool_t* for thread 0 and embed it into the rntm. We do
	// this up-front only so that we have the rntm_t.sba_pool field
	// initialized and ready for the global communicator creation below.
	bli_sba_rntm_set_pool( 0, array, rntm );

	// Set the packing block allocator field of the rntm. This will be
	// inherited by all of the child threads when they make local copies of
	// the rntm below.
	bli_pba_rntm_set_pba( rntm );

	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t* gl_comm = bli_thrcomm_create( rntm, n_threads );


	_Pragma( "omp parallel num_threads(n_threads)" )
	{
		// Create a thread-local copy of the master thread's rntm_t. This is
		// necessary since we want each thread to be able to track its own
		// small block pool_t as it executes down the function stack.
		rntm_t           rntm_l = *rntm;
		rntm_t* restrict rntm_p = &rntm_l;

		// Query the thread's id from OpenMP.
		const dim_t tid = omp_get_thread_num();

		// Check for a somewhat obscure OpenMP thread-mistmatch issue.
		bli_l3_thread_decorator_thread_check( n_threads, tid, gl_comm, rntm_p );

		// Use the thread id to access the appropriate pool_t* within the
		// array_t, and use it to set the sba_pool field within the rntm_t.
		// If the pool_t* element within the array_t is NULL, it will first
		// be allocated/initialized.
		bli_sba_rntm_set_pool( tid, array, rntm_p );

		thrinfo_t* thread = NULL;

		// Create the root node of the thread's thrinfo_t structure.
		bli_l3_sup_thrinfo_create_root( tid, gl_comm, rntm_p, &thread );

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

		// Free the current thread's thrinfo_t structure.
		bli_l3_sup_thrinfo_free( rntm_p, thread );
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called from the thread entry function).

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );
}

#endif

