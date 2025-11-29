/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2023, Southern Methodist University

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
#include <omp.h>
#endif

struct l3_decor_params_s
{
	const obj_t*   a;
	const obj_t*   b;
	const obj_t*   c;
	const cntx_t*  cntx;
	const cntl_t*  cntl;
	      rntm_t*  rntm;
	      array_t* array;
};
typedef struct l3_decor_params_s l3_decor_params_t;

static void bli_l3_thread_decorator_entry( thrcomm_t* gl_comm, dim_t tid, const void* data_void )
{
	const l3_decor_params_t* data    = data_void;

	const obj_t*             a       = data->a;
	const obj_t*             b       = data->b;
	const obj_t*             c       = data->c;
	const cntx_t*            cntx    = data->cntx;
	const cntl_t*            cntl    = data->cntl;
	      rntm_t*            rntm    = data->rntm;
	      array_t*           array   = data->array;

	bli_l3_thread_decorator_thread_check( gl_comm, rntm );

	// Create the root node of the current thread's thrinfo_t structure.
	// The root node is the *parent* of the node corresponding to the first
	// control tree node.
	thrinfo_t* thread = bli_l3_thrinfo_create( tid, gl_comm, array, rntm, cntl );

	bli_l3_int
	(
	  a,
	  b,
	  c,
	  cntx,
	  cntl,
	  thread
	);

	// Free the current thread's thrinfo_t structure.
	// NOTE: The barrier here is very important as it prevents memory being
	// released by the chief of some thread sub-group before its peers are done
	// using it. See PR #702 for more info [1].
	// [1] https://github.com/flame/blis/pull/702
	bli_thrinfo_barrier( thread );
	bli_thrinfo_free( thread );
}

void bli_l3_thread_decorator
     (
       const obj_t*   a,
       const obj_t*   b,
       const obj_t*   c,
       const cntx_t*  cntx,
       const cntl_t*  cntl,
       const rntm_t*  rntm
     )
{
	rntm_t rntm_l;
	if ( rntm != NULL ) rntm_l = *rntm;
	else bli_rntm_init_from_global( &rntm_l );

	// Set the number of ways for each loop, if needed, depending on what
	// kind of information is already stored in the rntm_t object.
	bli_rntm_factorize
	(
	  bli_obj_length( c ),
	  bli_obj_width( c ),
	  bli_obj_width( a ),
	  &rntm_l
	);

	// Query the threading implementation and the number of threads requested.
	timpl_t ti = bli_rntm_thread_impl( &rntm_l );
	dim_t   nt = bli_rntm_num_threads( &rntm_l );

#if 0
	printf( "(pre-opt) application requested rntm.thread_impl = %s\n",
	        ( ti == BLIS_SINGLE ? "single" :
	        ( ti == BLIS_OPENMP ? "openmp" : "pthreads" ) ) );
#endif

	if ( bli_error_checking_is_enabled() )
		bli_l3_thread_decorator_check( &rntm_l );

#ifdef BLIS_ENABLE_NT1_VIA_SINGLE
	if ( nt == 1 )
	{
		// An optimization. If the caller requests only one thread, force
		// the sequential level-3 thread decorator even if that means
		// overriding the caller's preferred threading implementation (as
		// communicated via the rntm_t).
		rntm_l = *rntm;
		ti = BLIS_SINGLE;
		bli_rntm_set_thread_impl( BLIS_SINGLE, &rntm_l );
		rntm = &rntm_l;
	}
#endif

	if ( 1 < nt && ti == BLIS_SINGLE )
	{
		// Here, we resolve conflicting information. The caller requested
		// a sequential threading implementation, but also requested more
		// than one thread. Here, we choose to favor the requested threading
		// implementation over the number of threads, and so reset all
		// parallelism parameters to 1.
		nt = 1;
		bli_rntm_set_ways_only( 1, 1, 1, 1, 1, &rntm_l );
		bli_rntm_set_num_threads_only( 1, &rntm_l );
	}

#if 0
	printf( "(post-opt) moving forward with rntm.thread_impl  = %s\n",
	        ( ti == BLIS_SINGLE ? "single" :
	        ( ti == BLIS_OPENMP ? "openmp" : "pthreads" ) ) );
#endif

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* array = bli_sba_checkout_array( nt );

	l3_decor_params_t params;
	params.a        = a;
	params.b        = b;
	params.c        = c;
	params.cntx     = cntx;
	params.cntl     = cntl;
	params.rntm     = &rntm_l;
	params.array    = array;

	// Launch the threads using the threading implementation specified by ti,
	// and use bli_l3_thread_decorator_entry() as their entry points. The
	// params struct will be passed along to each thread.
	bli_thread_launch( ti, nt, bli_l3_thread_decorator_entry, &params );

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );
}

void bli_l3_thread_decorator_check
     (
       const rntm_t* rntm
     )
{
	//err_t e_val;

	//e_val = bli_check_valid_thread_impl( bli_rntm_thread_impl( rntm ) );
	//bli_check_error_code( e_val );

	const timpl_t ti = bli_rntm_thread_impl( rntm );

	if (
#ifndef BLIS_ENABLE_OPENMP
	     ti == BLIS_OPENMP ||
#endif
#ifndef BLIS_ENABLE_PTHREADS
	     ti == BLIS_POSIX ||
#endif
#ifndef BLIS_ENABLE_HPX
	     ti == BLIS_HPX ||
#endif
	     FALSE
	   )
	{
		fprintf( stderr, "\n" );
		fprintf( stderr, "libblis: User requested threading implementation \"%s\", but that method is\n", bli_thread_get_thread_impl_str( ti ) );
		fprintf( stderr, "libblis: unavailable. Try reconfiguring BLIS with \"-t %s\" and recompiling.\n", bli_thread_get_thread_impl_str( ti ) );
		fprintf( stderr, "libblis: %s: line %d\n", __FILE__, ( int )__LINE__ );
		bli_abort();
	}
}

void bli_l3_thread_decorator_thread_check
     (
       thrcomm_t* gl_comm,
       rntm_t*    rntm
     )
{
#ifdef BLIS_ENABLE_OPENMP

	if ( bli_thrcomm_thread_impl( gl_comm ) != BLIS_OPENMP)
		return;

	dim_t n_threads_real = omp_get_num_threads();
	dim_t n_threads      = bli_thrcomm_num_threads( gl_comm );
	dim_t tid            = omp_get_thread_num();

	// Check if the number of OpenMP threads created within this parallel
	// region is different from the number of threads that were requested
	// of BLIS. This inequality may trigger when, for example, the
	// following conditions are satisfied:
	// - an application is executing an OpenMP parallel region in which
	//   BLIS is invoked,
	// - BLIS is configured for multithreading via OpenMP,
	// - OMP_NUM_THREADS = t > 1,
	// - the number of threads requested of BLIS (regardless of method)
	//   is p <= t,
	// - OpenMP nesting is disabled.
	// In this situation, the application spawns t threads. Each application
	// thread calls gemm (for example). Each gemm will attempt to spawn p
	// threads via OpenMP. However, since nesting is disabled, the OpenMP
	// implementation finds that t >= p threads are already spawned, and
	// thus it doesn't spawn *any* additional threads for each gemm.
	if ( n_threads_real != n_threads )
	{
		// If the number of threads active in the current region is not
		// equal to the number requested of BLIS, we then only continue
		// if the number of threads in the current region is 1. If, for
		// example, BLIS requested 4 threads but only got 3, then we
		// abort().
		if ( n_threads_real != 1 )
		{
			bli_print_msg( "A different number of threads was "
			               "created than was requested.",
			               __FILE__, __LINE__ );
			bli_abort();
		}

		if ( tid == 0 )
		{
			bli_thrcomm_init( BLIS_OPENMP, 1, gl_comm );
			bli_rntm_set_num_threads_only( 1, rntm );
			bli_rntm_set_ways_only( 1, 1, 1, 1, 1, rntm );
		}

		// Synchronize all threads and continue.
		_Pragma( "omp barrier" )
	}

#endif
}

