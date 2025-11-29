/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

void bli_thread_range_quad
     (
       const thrinfo_t* thread,
             doff_t     diagoff,
             uplo_t     uplo,
             dim_t      m,
             dim_t      n,
             dim_t      bf,
             bool       handle_edge_low,
             dim_t*     start,
             dim_t*     end,
             dim_t*     inc
     )
{
	const dim_t tid   = bli_thrinfo_work_id( thread );
	const dim_t jr_nt = bli_thrinfo_n_way( thread );
	const dim_t n_iter = n / bf + ( n % bf ? 1 : 0 );

#ifdef BLIS_ENABLE_JRIR_RR

	// Use round-robin (interleaved) partitioning of jr/ir loops.
	*start = tid;
	*end   = n_iter;
	*inc   = jr_nt;

#else // #elif defined( BLIS_ENABLE_JRIR_SLAB ) ||
	  //       defined( BLIS_ENABLE_JRIR_TLB  )

	// NOTE: While this cpp conditional branch applies to both _SLAB and _TLB
	// cases, this *function* should never be called when BLIS_ENABLE_JRIR_TLB
	// is defined, since the function is only called from macrokernels that were
	// designed for slab/rr partitioning.

	// If there is no parallelism in this loop, set the output variables
	// and return early.
	if ( jr_nt == 1 ) { *start = 0; *end = n_iter; *inc = 1; return; }

	// Local variables for the computed start, end, and increment.
	dim_t st, en, in;

	if ( bli_intersects_diag_n( diagoff, m, n ) )
	{
		// If the current submatrix intersects the diagonal, try to be
		// intelligent about how threads are assigned work by using the
		// quadratic partitioning function.

		bli_thread_range_weighted_sub
		(
		  thread, diagoff, uplo, uplo, m, n, bf,
		  handle_edge_low, &st, &en
		);
		in = bf;
	}
	else
	{
		// If the current submatrix does not intersect the diagonal, then we
		// are free to perform a uniform (and contiguous) slab partitioning.

		bli_thread_range_sub
		(
		  tid, jr_nt, n, bf,
		  handle_edge_low, &st, &en
		);
		in = bf;
	}

	// Convert the start and end column indices into micropanel indices by
	// dividing by the blocking factor (which, for the jr loop, is NR). If
	// either one yields a remainder, add an extra unit to the result. This
	// is necessary for situations where there are t threads with t-1 or
	// fewer micropanels of work, including an edge case. For example, if
	// t = 3 and n = 10 (with bf = NR = 8), then we want start and end for
	// each thread to be:
	//
	//                  column index           upanel index
	//   tid 0:  start, end =  0,  8  ->  start, end = 0, 1
	//   tid 1:  start, end =  8, 10  ->  start, end = 1, 2
	//   tid 2:  start, end = 10, 10  ->  start, end = 2, 2
	//
	// In this example, it's important that thread (tid) 2 gets no work, and
	// we express that by specifying start = end = n, which is a non-existent
	// column index.

	if ( st % bf == 0 ) *start = st / bf;
	else                *start = st / bf + 1;

	if ( en % bf == 0 ) *end = en / bf;
	else                *end = en / bf + 1;

	*inc = in / bf;

#endif
}
