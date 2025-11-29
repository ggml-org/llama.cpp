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

#if 0
void bli_thread_range_snake_jr
     (
       const thrinfo_t* thread,
             doff_t     diagoff,
             uplo_t     uplo,
             dim_t      n,
             dim_t      bf,
             bool       handle_edge_low,
             dim_t*     start,
             dim_t*     end,
             dim_t*     inc
     )
{
	// Use snake partitioning of jr loop.

	// NOTE: This function currently assumes that edge cases are handled
	// "high" and therefore ignores handle_edge_low. This is because the
	// function is only used by gemmt and friends (herk/her2k/syrk/syr2k).
	// These operations, unlike trmm/trmm3 and trsm, never require
	// low-range edge cases.

	const dim_t tid = bli_thrinfo_work_id( thread );
	const dim_t nt  = bli_thrinfo_n_way( thread );

	const dim_t n_left = n % bf;
	const dim_t n_iter = n / bf + ( n_left ? 1 : 0 );

	if ( bli_is_lower( uplo ) )
	{
		// Use the thrinfo_t work id as the thread's starting index.
		const dim_t st = tid;

		// This increment will be too big for some threads with only one unit
		// (NR columns, or an edge case) of work, but that's okay since all that
		// matters is that st + in >= en, which will cause that thread's jr loop
		// to not execute beyond the first iteration.
		const dim_t in = 2 * ( nt - tid ) - 1;

		      dim_t en = st + in + 1;

		// Don't let the thread's end index exceed n_iter.
		if ( n_iter < en ) en = n_iter;

		*start = st * bf;
		*end   = en * bf; // - ( bf - n_left );
		*inc   = in * bf;
	}
	else // if ( bli_is_upper( uplo ) )
	{
		      dim_t st = n_iter - 2 * nt + tid;

		const dim_t in = 2 * ( nt - tid ) - 1;

		      dim_t en = st + in + 1;

		#if 1
		// When nt exceeds half n_iter, some threads will only get one unit
		// (NR columns, or an edge case) of work. This manifests as st being
		// negative, and thus we need to move their start index to their other
		// assigned unit in the positive index range.
		if ( st < 0 ) st += in;

		// If the start index is *still* negative, which happens for some
		// threads when nt exceeds n_iter, then manually assign this thread
		// an empty index range.
		if ( st < 0 ) { st = 0; en = 0; }
		#else
		if ( 0 <= st + in ) { st += in; }
		else                { st = 0; en = 0; }
		#endif

		#if 0
		printf( "thread_range_snake_jr():  tid %d: sta end = %3d %3d %3d\n",
		        (int)tid, (int)(st), (int)(en), (int)(in) );
		#endif

		*start = st * bf;
		*end   = en * bf;
		*inc   = in * bf;
	}
}
#endif
