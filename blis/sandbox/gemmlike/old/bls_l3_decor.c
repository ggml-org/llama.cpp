/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

struct l3_sbx_decor_params_s
{
	l3sbxint_ft func;
	opid_t      family;
	obj_t*      alpha;
	obj_t*      a;
	obj_t*      b;
	obj_t*      beta;
	obj_t*      c;
	cntx_t*     cntx;
	rntm_t*     rntm;
	array_t*    array;
};
typedef struct l3_sbx_decor_params_s l3_sbx_decor_params_t;

static void bls_l3_thread_decorator_entry( thrcomm_t* gl_comm, dim_t tid, const void* data_void )
{
	const l3_sbx_decor_params_t* data   = data_void;

	l3sbxint_ft func   = data->func;
	opid_t      family = data->family;
	obj_t*      alpha  = data->alpha;
	obj_t*      a      = data->a;
	obj_t*      b      = data->b;
	obj_t*      beta   = data->beta;
	obj_t*      c      = data->c;
	cntx_t*     cntx   = data->cntx;
	rntm_t*     rntm   = data->rntm;
	array_t*    array  = data->array;

	( void )family;

	// Create the root node of the thread's thrinfo_t structure.
	pool_t*    sba_pool = bli_apool_array_elem( tid, array );
	thrinfo_t* thread   = bli_l3_sup_thrinfo_create( tid, gl_comm, sba_pool, rntm );

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
	bli_thrinfo_free( thread );
}

void bls_l3_thread_decorator
     (
       l3sbxint_ft func,
       opid_t      family,
       obj_t*      alpha,
       obj_t*      a,
       obj_t*      b,
       obj_t*      beta,
       obj_t*      c,
       cntx_t*     cntx,
       rntm_t*     rntm
     )
{
	rntm_t rntm_l = *rntm;

	// Query the threading implementation and the number of threads requested.
	timpl_t ti = bli_rntm_thread_impl( &rntm_l );
	dim_t   nt = bli_rntm_num_threads( &rntm_l );

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

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* array = bli_sba_checkout_array( nt );

	// Declare a params struct and embed within it all of the information
	// that is relevant to the computation.
	l3_sbx_decor_params_t params;
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

	// Launch the threads using the threading implementation specified by ti,
	// and use bli_l3_thread_decorator_entry() as their entry points. The
	// params struct will be passed along to each thread.
	bli_thread_launch( ti, nt, bls_l3_thread_decorator_entry, &params );

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );
}

