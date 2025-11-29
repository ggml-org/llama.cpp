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

// Initialize a function pointer array containing function addresses for
// each of the threading-specific level-3 thread decorators.

static l3ao_decor_ft l3ao_decor_fpa[ BLIS_NUM_THREAD_IMPLS ] =
{
	[BLIS_SINGLE] = bao_l3_thread_decorator_single,
	[BLIS_OPENMP] =
#if   defined(BLIS_ENABLE_OPENMP)
	                bao_l3_thread_decorator_openmp,
#elif defined(BLIS_ENABLE_PTHREADS)
	                NULL,
#else
	                NULL,
#endif
	[BLIS_POSIX]  =
#if   defined(BLIS_ENABLE_PTHREADS)
	                bao_l3_thread_decorator_pthreads,
#elif defined(BLIS_ENABLE_OPENMP)
	                NULL,
#else
	                NULL,
#endif
};

// Define a dispatcher that chooses a threading-specific function from the
// above function pointer array.

void bao_l3_thread_decorator
     (
       l3aoint_ft func,
       opid_t   family,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   d,
       obj_t*   b,
       obj_t*   beta,
       obj_t*   c,
       cntx_t*  cntx,
       rntm_t*  rntm
     )
{
	rntm_t rntm_l;

	// Query the threading implementation and the number of threads requested.
	timpl_t ti = bli_rntm_thread_impl( rntm );
	dim_t   nt = bli_rntm_num_threads( rntm );

	if ( bli_error_checking_is_enabled() )
		bao_l3_thread_decorator_check( rntm );

	if ( 1 < nt && ti == BLIS_SINGLE )
	{
		// Here, we resolve conflicting information. The caller requested
		// a sequential threading implementation, but also requested more
		// than one thread. Here, we choose to favor the requested threading
		// implementation over the number of threads, and so reset all
		// parallelism parameters to 1.
		rntm_l = *rntm;
		nt = 1;
		bli_rntm_set_ways_only( 1, 1, 1, 1, 1, &rntm_l );
		bli_rntm_set_num_threads_only( 1, &rntm_l );
		rntm = &rntm_l;
	}

	// Use the timpl_t value to index into the corresponding function address
	// from the function pointer array.
	const l3ao_decor_ft fp = l3ao_decor_fpa[ ti ];

	// Call the threading-specific decorator function.
	fp
	(
	  func,
	  family,
	  alpha,
	  a,
	  d,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm
	);
}

void bao_l3_thread_decorator_check
     (
       rntm_t* rntm
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
	    FALSE
	   )
	{
		fprintf( stderr, "\n" );
		fprintf( stderr, "libblis: User requested threading implementation \"%s\", but that method is\n", ( ti == BLIS_OPENMP ? "openmp" : "pthreads" ) );
		fprintf( stderr, "libblis: unavailable. Try reconfiguring BLIS with \"-t %s\" and recompiling.\n", ( ti == BLIS_OPENMP ? "openmp" : "pthreads" ) );
		fprintf( stderr, "libblis: %s: line %d\n", __FILE__, ( int )__LINE__ );
		bli_abort();
	}
}

