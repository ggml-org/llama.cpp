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
#include "blix.h"

// This code is enabled only when multithreading is enabled via OpenMP.
#ifdef BLIS_ENABLE_OPENMP

#if 0
void blx_gemm_thread
     (
       gemmint_t func,
       opid_t    family,
       obj_t*    a,
       obj_t*    b,
       obj_t*    c,
       cntx_t*   cntx,
       rntm_t*   rntm,
       cntl_t*   cntl
     )
{
	// Query the total number of threads from the context.
	dim_t       n_threads = bli_rntm_num_threads( rntm );

	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t*  gl_comm   = bli_thrcomm_create( n_threads );

	_Pragma( "omp parallel num_threads(n_threads)" )
	{
		dim_t      id = omp_get_thread_num();

		obj_t      a_t, b_t, c_t;
		cntl_t*    cntl_use;
		thrinfo_t* thread;

		// Alias thread-local copies of A, B, and C. These will be the objects
		// we pass down the algorithmic function stack. Making thread-local
		// alaises IS ABSOLUTELY IMPORTANT and MUST BE DONE because each thread
		// will read the schemas from A and B and then reset the schemas to
		// their expected unpacked state (in blx_l3_cntl_create_if()).
		bli_obj_alias_to( a, &a_t );
		bli_obj_alias_to( b, &b_t );
		bli_obj_alias_to( c, &c_t );

		// Create a default control tree for the operation, if needed.
		blx_l3_cntl_create_if( family, &a_t, &b_t, &c_t, cntl, &cntl_use );

		// Create the root node of the current thread's thrinfo_t structure.
		bli_l3_thrinfo_create_root( id, gl_comm, rntm, cntl_use, &thread );

		func
		(
		  &a_t,
		  &b_t,
		  &c_t,
		  cntx,
		  rntm,
		  cntl_use,
		  thread
		);

		// Free the control tree, if one was created locally.
		blx_l3_cntl_free_if( &a_t, &b_t, &c_t, cntl, cntl_use, thread );

		// Free the current thread's thrinfo_t structure.
		bli_l3_thrinfo_free( thread );
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called above).
}
#endif
void blx_gemm_thread
     (
       gemmint_t  func,
       opid_t     family,
       obj_t*     alpha,
       obj_t*     a,
       obj_t*     b,
       obj_t*     beta,
       obj_t*     c,
       cntx_t*    cntx,
       rntm_t*    rntm,
       cntl_t*    cntl
     )
{
	// This is part of a hack to support mixed domain in bli_gemm_front().
	// Sometimes we need to specify a non-standard schema for A and B, and
	// we decided to transmit them via the schema field in the obj_t's
	// rather than pass them in as function parameters. Once the values
	// have been read, we immediately reset them back to their expected
	// values for unpacked objects.
	pack_t schema_a = bli_obj_pack_schema( a );
	pack_t schema_b = bli_obj_pack_schema( b );
	bli_obj_set_pack_schema( BLIS_NOT_PACKED, a );
	bli_obj_set_pack_schema( BLIS_NOT_PACKED, b );

	// Query the total number of threads from the rntm_t object.
	const dim_t n_threads = bli_rntm_num_threads( rntm );

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* restrict array = bli_sba_checkout_array( n_threads );

	// Access the pool_t* for thread 0 and embed it into the rntm. We do
	// this up-front only so that we have the rntm_t.sba_pool field
	// initialized and ready for the global communicator creation below.
	bli_sba_rntm_set_pool( 0, array, rntm );

	// Set the packing block allocator field of the rntm. This will be
	// inherited by all of the child threads when they make local copies of
	// the rntm below.
	bli_pba_rntm_set_pba( rntm );

	// Allocate a global communicator for the root thrinfo_t structures.
	thrcomm_t* restrict gl_comm = bli_thrcomm_create( rntm, n_threads );


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
		//bli_l3_thread_decorator_thread_check( n_threads, tid, gl_comm, rntm_p );

		// Use the thread id to access the appropriate pool_t* within the
		// array_t, and use it to set the sba_pool field within the rntm_t.
		// If the pool_t* element within the array_t is NULL, it will first
		// be allocated/initialized.
		bli_sba_rntm_set_pool( tid, array, rntm_p );


		obj_t      a_t, b_t, c_t;
		cntl_t*    cntl_use;
		thrinfo_t* thread;

		// Alias thread-local copies of A, B, and C. These will be the objects
		// we pass down the algorithmic function stack. Making thread-local
		// aliases is highly recommended in case a thread needs to change any
		// of the properties of an object without affecting other threads'
		// objects.
		bli_obj_alias_to( a, &a_t );
		bli_obj_alias_to( b, &b_t );
		bli_obj_alias_to( c, &c_t );

		// Create a default control tree for the operation, if needed.
		blx_l3_cntl_create_if( family, schema_a, schema_b,
		                       &a_t, &b_t, &c_t, rntm_p, cntl, &cntl_use );

		// Create the root node of the current thread's thrinfo_t structure.
		blx_l3_thrinfo_create_root( tid, gl_comm, rntm_p, cntl_use, &thread );

		func
		(
		  alpha,
		  &a_t,
		  &b_t,
		  beta,
		  &c_t,
		  cntx,
		  rntm_p,
		  cntl_use,
		  thread
		);

		// Free the thread's local control tree.
		blx_l3_cntl_free( rntm_p, cntl_use, thread );

		// Free the current thread's thrinfo_t structure.
		bli_l3_thrinfo_free( rntm_p, thread );
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called above).

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );
}



#endif

#ifdef BLIS_ENABLE_PTHREADS
#error "Sandbox does not yet implement pthreads."
#endif

// This code is enabled only when multithreading is disabled.
#ifndef BLIS_ENABLE_MULTITHREADING

void blx_gemm_thread
     (
       gemmint_t func,
       opid_t    family,
       obj_t*    a,
       obj_t*    b,
       obj_t*    c,
       cntx_t*   cntx,
       rntm_t*   rntm,
       cntl_t*   cntl
     )
{
	// For sequential execution, we use only one thread.
	dim_t      n_threads = 1;
	dim_t      id        = 0;

	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t* gl_comm   = bli_thrcomm_create( n_threads );

	cntl_t*    cntl_use;
	thrinfo_t* thread;

	// Create a default control tree for the operation, if needed.
	blx_l3_cntl_create_if( family, a, b, c, cntl, &cntl_use );

	// Create the root node of the thread's thrinfo_t structure.
	bli_l3_thrinfo_create_root( id, gl_comm, rntm, cntl_use, &thread );

	func
	(
	  a,
	  b,
	  c,
	  cntx,
	  rntm,
	  cntl_use,
	  thread
	);

	// Free the control tree, if one was created locally.
	blx_l3_cntl_free_if( a, b, c, cntl, cntl_use, thread );

	// Free the current thread's thrinfo_t structure.
	bli_l3_thrinfo_free( thread );

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called above).
}

#endif

