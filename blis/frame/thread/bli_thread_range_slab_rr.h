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

#ifndef BLIS_THREAD_RANGE_SLAB_RR_H
#define BLIS_THREAD_RANGE_SLAB_RR_H

BLIS_INLINE void bli_thread_range_rr
     (
       dim_t  tid,
       dim_t  nt,
       dim_t  n,
       dim_t  bf,
       bool   handle_edge_low,
       dim_t* start,
       dim_t* end,
       dim_t* inc
     )
{
	const dim_t n_iter = n / bf + ( n % bf ? 1 : 0 );

	// Use round-robin (interleaved) partitioning of jr/ir loops.
	*start = tid;
	*end   = n_iter;
	*inc   = nt;
}

BLIS_INLINE void bli_thread_range_sl
     (
       dim_t  work_id,
       dim_t  n_way,
       dim_t  n,
       dim_t  bf,
       bool   handle_edge_low,
       dim_t* start,
       dim_t* end,
       dim_t* inc
     )
{
	// Use contiguous slab partitioning of jr/ir loops.
	bli_thread_range_sub( work_id, n_way, n, bf, handle_edge_low, start, end );
	*inc = 1;
}

BLIS_INLINE void bli_thread_range_slrr
     (
       dim_t  work_id,
       dim_t  n_way,
       dim_t  n,
       dim_t  bf,
       bool   handle_edge_low,
       dim_t* start,
       dim_t* end,
       dim_t* inc
     )
{
	// Define a general-purpose slab/rr function whose definition depends on
	// whether slab or round-robin partitioning was requested at configure-time.
	// Note that this function also uses the slab code path when tlb is enabled.
	// If this is ever changed, make sure to change bli_is_my_iter() since they
	// are used together by packm.

#ifdef BLIS_ENABLE_JRIR_RR
	bli_thread_range_rr( work_id, n_way, bf, handle_edge_low, start, end, inc );
#else // ifdef ( _SLAB || _TLB )
	bli_thread_range_sl( work_id, n_way, n, bf, handle_edge_low, start, end, inc );
#endif
}

// -----------------------------------------------------------------------------

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
     );

#endif

