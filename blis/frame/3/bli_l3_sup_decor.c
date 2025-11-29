/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2024, Southern Methodist University

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

struct l3_sup_decor_params_s
{
	      l3supint_ft func;
	      opid_t      family;
	const obj_t*      alpha;
	const obj_t*      a;
	const obj_t*      b;
	const obj_t*      beta;
	const obj_t*      c;
	const cntx_t*     cntx;
	      rntm_t*     rntm;
	      array_t*    array;
};
typedef struct l3_sup_decor_params_s l3_sup_decor_params_t;

static void bli_l3_sup_thread_decorator_entry( thrcomm_t* gl_comm, dim_t tid, const void* data_void )
{
	const l3_sup_decor_params_t* data    = data_void;

	const l3supint_ft            func    = data->func;
	const opid_t                 family  = data->family;
	const obj_t*                 alpha   = data->alpha;
	const obj_t*                 a       = data->a;
	const obj_t*                 b       = data->b;
	const obj_t*                 beta    = data->beta;
	const obj_t*                 c       = data->c;
	const cntx_t*                cntx    = data->cntx;
	      rntm_t*                rntm    = data->rntm;
	      array_t*               array   = data->array;

	( void )family;

	bli_l3_thread_decorator_thread_check( gl_comm, rntm );

	// Create the root node of the thread's thrinfo_t structure.
	pool_t*    pool   = bli_sba_array_elem( tid, array );
	thrinfo_t* thread = bli_l3_sup_thrinfo_create( tid, gl_comm, pool, rntm );

	func
	(
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm,
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

err_t bli_l3_sup_thread_decorator
     (
             l3supint_ft func,
             opid_t   family,
       const obj_t*   alpha,
       const obj_t*   a,
       const obj_t*   b,
       const obj_t*   beta,
       const obj_t*   c,
       const cntx_t*  cntx,
       const rntm_t*  rntm
     )
{
	rntm_t rntm_l = *rntm;

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

	l3_sup_decor_params_t params;
	params.func   = func;
	params.family = family;
	params.alpha  = alpha;
	params.a      = a;
	params.b      = b;
	params.beta   = beta;
	params.c      = c;
	params.cntx   = cntx;
	params.rntm   = &rntm_l;
	params.array  = array;

	bli_thread_launch( ti, nt, bli_l3_sup_thread_decorator_entry, &params );

	bli_sba_checkin_array( array );

	return BLIS_SUCCESS;
}

