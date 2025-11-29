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

void bli_thread_range_sub
     (
       dim_t  work_id,
       dim_t  n_way,
       dim_t  n,
       dim_t  bf,
       bool   handle_edge_low,
       dim_t* start,
       dim_t* end
     )
{
	if ( n_way == 1 ) { *start = 0; *end = n; return; }

	dim_t      all_start  = 0;
	dim_t      all_end    = n;

	dim_t      size       = all_end - all_start;

	dim_t      n_bf_whole = size / bf;
	dim_t      n_bf_left  = size % bf;

	dim_t      n_bf_lo    = n_bf_whole / n_way;
	dim_t      n_bf_hi    = n_bf_whole / n_way;

	// In this function, we partition the space between all_start and
	// all_end into n_way partitions, each a multiple of block_factor
	// with the exception of the one partition that recieves the
	// "edge" case (if applicable).
	//
	// Here are examples of various thread partitionings, in units of
	// the block_factor, when n_way = 4. (A '+' indicates the thread
	// that receives the leftover edge case (ie: n_bf_left extra
	// rows/columns in its sub-range).
	//                                        (all_start ... all_end)
	// n_bf_whole  _left  hel  n_th_lo  _hi   thr0  thr1  thr2  thr3
	//         12     =0    f        0    4      3     3     3     3
	//         12     >0    f        0    4      3     3     3     3+
	//         13     >0    f        1    3      4     3     3     3+
	//         14     >0    f        2    2      4     4     3     3+
	//         15     >0    f        3    1      4     4     4     3+
	//         15     =0    f        3    1      4     4     4     3
	//
	//         12     =0    t        4    0      3     3     3     3
	//         12     >0    t        4    0      3+    3     3     3
	//         13     >0    t        3    1      3+    3     3     4
	//         14     >0    t        2    2      3+    3     4     4
	//         15     >0    t        1    3      3+    4     4     4
	//         15     =0    t        1    3      3     4     4     4

	// As indicated by the table above, load is balanced as equally
	// as possible, even in the presence of an edge case.

	// First, we must differentiate between cases where the leftover
	// "edge" case (n_bf_left) should be allocated to a thread partition
	// at the low end of the index range or the high end.

	if ( handle_edge_low == FALSE )
	{
		// Notice that if all threads receive the same number of
		// block_factors, those threads are considered "high" and
		// the "low" thread group is empty.
		dim_t n_th_lo = n_bf_whole % n_way;
		//dim_t n_th_hi = n_way - n_th_lo;

		// If some partitions must have more block_factors than others
		// assign the slightly larger partitions to lower index threads.
		if ( n_th_lo != 0 ) n_bf_lo += 1;

		// Compute the actual widths (in units of rows/columns) of
		// individual threads in the low and high groups.
		dim_t size_lo = n_bf_lo * bf;
		dim_t size_hi = n_bf_hi * bf;

		// Precompute the starting indices of the low and high groups.
		dim_t lo_start = all_start;
		dim_t hi_start = all_start + n_th_lo * size_lo;

		// Compute the start and end of individual threads' ranges
		// as a function of their work_ids and also the group to which
		// they belong (low or high).
		if ( work_id < n_th_lo )
		{
			*start = lo_start + (work_id  ) * size_lo;
			*end   = lo_start + (work_id+1) * size_lo;
		}
		else // if ( n_th_lo <= work_id )
		{
			*start = hi_start + (work_id-n_th_lo  ) * size_hi;
			*end   = hi_start + (work_id-n_th_lo+1) * size_hi;

			// Since the edge case is being allocated to the high
			// end of the index range, we have to advance the last
			// thread's end.
			if ( work_id == n_way - 1 ) *end += n_bf_left;
		}
	}
	else // if ( handle_edge_low == TRUE )
	{
		// Notice that if all threads receive the same number of
		// block_factors, those threads are considered "low" and
		// the "high" thread group is empty.
		dim_t n_th_hi = n_bf_whole % n_way;
		dim_t n_th_lo = n_way - n_th_hi;

		// If some partitions must have more block_factors than others
		// assign the slightly larger partitions to higher index threads.
		if ( n_th_hi != 0 ) n_bf_hi += 1;

		// Compute the actual widths (in units of rows/columns) of
		// individual threads in the low and high groups.
		dim_t size_lo = n_bf_lo * bf;
		dim_t size_hi = n_bf_hi * bf;

		// Precompute the starting indices of the low and high groups.
		dim_t lo_start = all_start;
		dim_t hi_start = all_start + n_th_lo * size_lo
		                           + n_bf_left;

		// Compute the start and end of individual threads' ranges
		// as a function of their work_ids and also the group to which
		// they belong (low or high).
		if ( work_id < n_th_lo )
		{
			*start = lo_start + (work_id  ) * size_lo;
			*end   = lo_start + (work_id+1) * size_lo;

			// Since the edge case is being allocated to the low
			// end of the index range, we have to advance the
			// starts/ends accordingly.
			if ( work_id == 0 )   *end   += n_bf_left;
			else                { *start += n_bf_left;
			                      *end   += n_bf_left; }
		}
		else // if ( n_th_lo <= work_id )
		{
			*start = hi_start + (work_id-n_th_lo  ) * size_hi;
			*end   = hi_start + (work_id-n_th_lo+1) * size_hi;
		}
	}
}

// -----------------------------------------------------------------------------

dim_t bli_thread_range_width_l
     (
       doff_t diagoff_j,
       dim_t  m,
       dim_t  n_j,
       dim_t  j,
       dim_t  n_way,
       dim_t  bf,
       dim_t  bf_left,
       double area_per_thr,
       bool   handle_edge_low
     )
{
	dim_t width;

	// In this function, we assume that we are somewhere in the process of
	// partitioning an m x n lower-stored region (with arbitrary diagonal
	// offset) n_ways along the n dimension (into column panels). The value
	// j identifies the left-to-right subpartition index (from 0 to n_way-1)
	// of the subpartition whose width we are about to compute using the
	// area per thread determined by the caller. n_j is the number of
	// columns in the remaining region of the matrix being partitioned,
	// and diagoff_j is that region's diagonal offset.

	// If this is the last subpartition, the width is simply equal to n_j.
	// Note that this statement handles cases where the "edge case" (if
	// one exists) is assigned to the high end of the index range (ie:
	// handle_edge_low == FALSE).
	if ( j == n_way - 1 ) return n_j;

	// At this point, we know there are at least two subpartitions left.
	// We also know that IF the submatrix contains a completely dense
	// rectangular submatrix, it will occur BEFORE the triangular (or
	// trapezoidal) part.

	// Here, we implement a somewhat minor load balancing optimization
	// that ends up getting employed only for relatively small matrices.
	// First, recall that all subpartition widths will be some multiple
	// of the blocking factor bf, except perhaps either the first or last
	// subpartition, which will receive the edge case, if it exists.
	// Also recall that j represents the current thread (or thread group,
	// or "caucus") for which we are computing a subpartition width.
	// If n_j is sufficiently small that we can only allocate bf columns
	// to each of the remaining threads, then we set the width to bf. We
	// do not allow the subpartition width to be less than bf, so, under
	// some conditions, if n_j is small enough, some of the reamining
	// threads may not get any work. For the purposes of this lower bound
	// on work (ie: width >= bf), we allow the edge case to count as a
	// "full" set of bf columns.
	{
		dim_t n_j_bf = n_j / bf + ( bf_left > 0 ? 1 : 0 );

		if ( n_j_bf <= n_way - j )
		{
			if ( j == 0 && handle_edge_low )
				width = ( bf_left > 0 ? bf_left : bf );
			else
				width = bf;

			// Make sure that the width does not exceed n_j. This would
			// occur if and when n_j_bf < n_way - j; that is, when the
			// matrix being partitioned is sufficiently small relative to
			// n_way such that there is not even enough work for every
			// (remaining) thread to get bf (or bf_left) columns. The
			// net effect of this safeguard is that some threads may get
			// assigned empty ranges (ie: no work), which of course must
			// happen in some situations.
			if ( width > n_j ) width = n_j;

			return width;
		}
	}

	// This block computes the width assuming that we are entirely within
	// a dense rectangle that precedes the triangular (or trapezoidal)
	// part.
	{
		// First compute the width of the current panel under the
		// assumption that the diagonal offset would not intersect.
		width = ( dim_t )bli_round( ( double )area_per_thr / ( double )m );

		// Adjust the width, if necessary. Specifically, we may need
		// to allocate the edge case to the first subpartition, if
		// requested; otherwise, we just need to ensure that the
		// subpartition is a multiple of the blocking factor.
		if ( j == 0 && handle_edge_low )
		{
			if ( width % bf != bf_left ) width += bf_left - ( width % bf );
		}
		else // if interior case
		{
			// Round up to the next multiple of the blocking factor.
			//if ( width % bf != 0       ) width += bf      - ( width % bf );
			// Round to the nearest multiple of the blocking factor.
			if ( width % bf != 0       ) width = bli_round_to_mult( width, bf );
		}
	}

	// We need to recompute width if the panel, according to the width
	// as currently computed, would intersect the diagonal.
	if ( diagoff_j < width )
	{
		dim_t offm_inc, offn_inc;

		// Prune away the unstored region above the diagonal, if it exists.
		// Note that the entire region was pruned initially, so we know that
		// we don't need to try to prune the right side. (Also, we discard
		// the offset deltas since we don't need to actually index into the
		// subpartition.)
		bli_prune_unstored_region_top_l( &diagoff_j, &m, &n_j, &offm_inc );
		//bli_prune_unstored_region_right_l( &diagoff_j, &m, &n_j, &offn_inc );

		// We don't need offm_inc, offn_inc here. These statements should
		// prevent compiler warnings.
		( void )offm_inc;
		( void )offn_inc;

		// Prepare to solve a quadratic equation to find the width of the
		// current (jth) subpartition given the m dimension, diagonal offset,
		// and area.
		// NOTE: We know that the +/- in the quadratic formula must be a +
		// here because we know that the desired solution (the subpartition
		// width) will be smaller than (m + diagoff), not larger. If you
		// don't believe me, draw a picture!
		const double a = -0.5;
		const double b = ( double )m + ( double )diagoff_j + 0.5;
		const double c = -0.5 * (   ( double )diagoff_j *
		                          ( ( double )diagoff_j + 1.0 )
		                        ) - area_per_thr;
		const double r = b * b - 4.0 * a * c;

		// If the quadratic solution is not imaginary, round it and use that
		// as our width (but make sure it didn't round to zero). Otherwise,
		// discard the quadratic solution and leave width, as previously
		// computed, unchanged.
		if ( r >= 0.0 )
		{
			const double x = ( -b + sqrt( r ) ) / ( 2.0 * a );

			width = ( dim_t )bli_round( x );
			if ( width == 0 ) width = 1;
		}

		// Adjust the width, if necessary.
		if ( j == 0 && handle_edge_low )
		{
			if ( width % bf != bf_left ) width += bf_left - ( width % bf );
		}
		else // if interior case
		{
			// Round up to the next multiple of the blocking factor.
			//if ( width % bf != 0       ) width += bf      - ( width % bf );
			// Round to the nearest multiple of the blocking factor.
			if ( width % bf != 0       ) width = bli_round_to_mult( width, bf );
		}
	}

	// Make sure that the width, after being adjusted, does not cause the
	// subpartition to exceed n_j.
	if ( width > n_j ) width = n_j;

	return width;
}

siz_t bli_find_area_trap_l
     (
       doff_t diagoff,
       dim_t  m,
       dim_t  n,
       dim_t  bf
     )
{
	dim_t  offm_inc = 0;
	dim_t  offn_inc = 0;
	double utri_area;
	double blktri_area;

	// Prune away any rectangular region above where the diagonal
	// intersects the left edge of the subpartition, if it exists.
	bli_prune_unstored_region_top_l( &diagoff, &m, &n, &offm_inc );

	// Prune away any rectangular region to the right of where the
	// diagonal intersects the bottom edge of the subpartition, if
	// it exists. (This shouldn't ever be needed, since the caller
	// would presumably have already performed rightward pruning,
	// but it's here just in case.)
	//bli_prune_unstored_region_right_l( &diagoff, &m, &n, &offn_inc );

	( void )offm_inc;
	( void )offn_inc;

	// Compute the area of the empty triangle so we can subtract it
	// from the area of the rectangle that bounds the subpartition.
	if ( bli_intersects_diag_n( diagoff, m, n ) )
	{
		double tri_dim = ( double )( n - diagoff - 1 );
		       tri_dim = bli_min( tri_dim, m - 1 );

		utri_area   = tri_dim * ( tri_dim + 1.0 ) / 2.0;
		blktri_area = tri_dim * ( bf      - 1.0 ) / 2.0;
	}
	else
	{
		// If the diagonal does not intersect the trapezoid, then
		// we can compute the area as a simple rectangle.
		utri_area   = 0.0;
		blktri_area = 0.0;
	}

	double area = ( double )m * ( double )n - utri_area + blktri_area;

	return ( siz_t )area;
}

// -----------------------------------------------------------------------------

siz_t bli_thread_range_weighted_sub
     (
       const thrinfo_t* thread,
             doff_t     diagoff,
             uplo_t     uplo,
             uplo_t     uplo_orig,
             dim_t      m,
             dim_t      n,
             dim_t      bf,
             bool       handle_edge_low,
             dim_t*     j_start_thr,
             dim_t*     j_end_thr
     )
{
	dim_t      n_way   = bli_thrinfo_n_way( thread );
	dim_t      my_id   = bli_thrinfo_work_id( thread );

	dim_t      bf_left = n % bf;

	dim_t      offm_inc, offn_inc;

	siz_t      area = 0;

	// In this function, we assume that the caller has already determined
	// that (a) the diagonal intersects the submatrix, and (b) the submatrix
	// is either lower- or upper-stored.

	if ( bli_is_lower( uplo ) )
	{
		#if 0
		if ( n_way > 1 )
		printf( "thread_range_weighted_sub(): tid %d: m n = %3d %3d do %d (lower)\n",
		        (int)my_id, (int)(m), (int)(n), (int)(diagoff) );
		#endif

		// Prune away the unstored region above the diagonal, if it exists,
		// and then to the right of where the diagonal intersects the bottom,
		// if it exists. (Also, we discard the offset deltas since we don't
		// need to actually index into the subpartition.)
		bli_prune_unstored_region_top_l( &diagoff, &m, &n, &offm_inc );

		if ( !handle_edge_low )
		{
			// This branch handles the following two cases:
			// - note: Edge case microtiles are marked as 'e'.
			//
			// uplo_orig = lower       | uplo = lower
			// handle edge high (orig) | handle edge high
			//
			//     x x x x x x x              x x x x x x x
			//     x x x x x x x x            x x x x x x x x
			//     x x x x x x x x x      ->  x x x x x x x x x
			//     x x x x x x x x x x        x x x x x x x x x x
			//     x x x x x x x x x x e      x x x x x x x x x x e
			//     x x x x x x x x x x e      x x x x x x x x x x e
			//
			// uplo_orig = upper       | uplo = lower
			// handle edge low  (orig) | handle edge high
			//
			//     e x x x x x x x x x x      x x x x x x x
			//     e x x x x x x x x x x      x x x x x x x x
			//       x x x x x x x x x x  ->  x x x x x x x x x
			//         x x x x x x x x x      x x x x x x x x x x
			//           x x x x x x x x      x x x x x x x x x x e
			//             x x x x x x x      x x x x x x x x x x e

			// If the edge case is being handled "high", then we can employ this
			// simple macro for pruning the region to the right of where the
			// diagonal intersets the right side of the submatrix (which amounts
			// to adjusting the n dimension).
			bli_prune_unstored_region_right_l( &diagoff, &m, &n, &offn_inc );
		}
		else // if ( handle_edge_low )
		{
			// This branch handles the following two cases:
			//
			// uplo_orig = upper       | uplo = lower
			// handle edge high (orig) | handle edge low
			//
			//     x x x x x x x x x x e      e x x x x x x
			//     x x x x x x x x x x e      e x x x x x x x
			//       x x x x x x x x x e  ->  e x x x x x x x x
			//         x x x x x x x x e      e x x x x x x x x x
			//           x x x x x x x e      e x x x x x x x x x x
			//             x x x x x x e      e x x x x x x x x x x
			//
			// uplo_orig = lower       | uplo = lower
			// handle edge low  (orig) | handle edge low
			//
			//     e x x x x x x              e x x x x x x
			//     e x x x x x x x            e x x x x x x x
			//     e x x x x x x x x      ->  e x x x x x x x x
			//     e x x x x x x x x x        e x x x x x x x x x
			//     e x x x x x x x x x x      e x x x x x x x x x x
			//     e x x x x x x x x x x      e x x x x x x x x x x

			// If the edge case is being handled "low", then we have to be more
			// careful. The problem can be seen in certain situations when we're
			// actually computing the weighted ranges for an upper-stored
			// subpartition whose (a) diagonal offset is positive (though will
			// always be less than NR), (b) right-side edge case exists, and (c)
			// sum of (a) and (b) is less than NR. This is a problem because the
			// upcoming loop that iterates over/ bli_thread_range_width_l()
			// doesn't realize that the offsets associated with (a) and (b)
			// belong on two separate columns of microtiles. If we naively use
			// bli_prune_unstored_region_right_l() when handle_edge_low == TRUE,
			// the loop over bli_thread_range_width_l() will only "see" p-1
			// IR-iterations of work to assign to threads when there are
			// actually p micropanels.

			const dim_t n_inner = ( diagoff + bli_min( m, n - diagoff ) - bf_left );

			const dim_t n_bf_iter_br = n_inner / bf;
			const dim_t n_bf_left_br = n_inner % bf;
			const dim_t n_bf_br = ( bf_left > 0 ? 1 : 0 ) +
			                        n_bf_iter_br +
			                      ( n_bf_left_br > 0 ? 1 : 0 );

			// Compute the number of extra columns that were included in n_bf_br
			// as a result of including a full micropanel for the part of the
			// submatrix that contains bf_left columns. For example, if bf = 16
			// and bf_left = 4, then bf_extra = 12. But if bf_left = 0, then we
			// didn't include any extra columns.
			const dim_t bf_extra = ( bf_left > 0 ? bf - bf_left : 0 );

			// Subtract off bf_extra from n_bf_br to arrive at the "true" value
			// of n that we'll use going forward.
			n = n_bf_br * bf - bf_extra;

			#if 0
			if ( n_way > 1 )
			{
				//printf( "thread_range_weighted_sub(): tid %d: _iter _left = %3d %3d (lower1)\n",
				//		(int)my_id, (int)n_bf_iter_br, (int)n_bf_left_br );
				printf( "thread_range_weighted_sub(): tid %d: m n = %3d %3d do %d (lower2)\n",
						(int)my_id, (int)(m), (int)(n), (int)(diagoff) );
			}
			#endif
		}

		// We don't need offm_inc, offn_inc here. These statements should
		// prevent compiler warnings.
		( void )offm_inc;
		( void )offn_inc;

		// Now that pruning has taken place, we know that diagoff >= 0.

		// Compute the total area of the submatrix, accounting for the
		// location of the diagonal. This is done by computing the area in
		// the strictly upper triangle, subtracting it off the area of the
		// full rectangle, and then adding the missing strictly upper
		// triangles of the bf x bf blocks along the diagonal.
		double tri_dim     = ( double )( n - diagoff - 1 );
		       tri_dim     = bli_min( tri_dim, m - 1 );
		double utri_area   = tri_dim * ( tri_dim + 1.0 ) / 2.0;

		// Note that the expression below is the simplified form of:
		//   blktri_area = ( tri_dim / bf ) * bf * ( bf - 1.0 ) / 2.0;
		double blktri_area = tri_dim * ( bf - 1.0 ) / 2.0;

		// Compute the area of the region to the right of where the diagonal
		// intersects the bottom edge of the submatrix. If it instead intersects
		// the right edge (or the bottom-right corner), then this region does
		// not exist and so its area is explicitly set to zero.
		double beyondtri_dim = n - diagoff - m;
		double beyondtri_area;
		if ( 0 < beyondtri_dim ) beyondtri_area = beyondtri_dim * m;
		else                     beyondtri_area = 0.0;

		// Here, we try to account for the added cost of computing columns of
		// microtiles that intersect the diagonal. This is rather difficult to
		// model, but this is partly due to the way non-square microtiles map
		// onto the matrix relative to the diagonal, as well as additional
		// overhead incurred from (potentially) computing with less-than-full
		// columns of microtiles (i.e., columns for which diagoff_j < 0).
		// Note that higher values for blktri_area have the net effect of
		// increasing the relative size of slabs that share little or no overlap
		// with the diagonal region. this is because it slightly increases the
		// total area computation below, which in turn increases the area
		// targeted by each thread/group earlier in the thread range, which
		// for lower trapezoidal submatrices, corresponds to the regular
		// rectangular region that precedes the diagonal part (if such a
		// rectangular region exists).
		blktri_area *= 1.5;
		//blktri_area = 0.0;

		double area_total  = ( double )m * ( double )n - utri_area + blktri_area
		                                               - beyondtri_area;

		// Divide the computed area by the number of ways of parallelism.
		double area_per_thr = area_total / ( double )n_way;


		// Initialize some variables prior to the loop: the offset to the
		// current subpartition, the remainder of the n dimension, and
		// the diagonal offset of the current subpartition.
		dim_t  off_j     = 0;
		doff_t diagoff_j = diagoff;
		dim_t  n_left    = n;

		#if 0
		printf( "thread_range_weighted_sub(): tid %d: n_left = %3d       (lower4)\n",
		        (int)my_id, (int)(n_left) );
		#endif

		// Iterate over the subpartition indices corresponding to each
		// thread/caucus participating in the n_way parallelism.
		for ( dim_t j = 0; j < n_way; ++j )
		{
			// Compute the width of the jth subpartition, taking the
			// current diagonal offset into account, if needed.
			dim_t width_j
			=
			bli_thread_range_width_l
			(
			  diagoff_j, m, n_left,
			  j, n_way,
			  bf, bf_left,
			  area_per_thr,
			  handle_edge_low
			);

			#if 0
			if ( n_way > 1 )
			printf( "thread_range_weighted_sub(): tid %d: width_j = %d doff_j = %d\n",
			        (int)my_id, (int)width_j, (int)diagoff_j );
			#endif

			// If the current thread belongs to caucus j, this is his
			// subpartition. So we compute the implied index range and
			// end our search.
			#if 0
			// An alternate way of assigning work to threads such that regions
			// are assigned to threads left to right *after* accounting for the
			// fact that we recycle the same lower-trapezoidal code to also
			// compute the upper-trapezoidal case.
			bool is_my_range;
			if ( bli_is_lower( uplo_orig ) ) is_my_range = ( j ==         my_id     );
			else                             is_my_range = ( j == n_way - my_id - 1 );
			#else
			bool is_my_range = ( j == my_id );
			#endif

			if ( is_my_range )
			{
				*j_start_thr = off_j;
				*j_end_thr   = off_j + width_j;

				#if 0
				if ( n_way > 1 )
				printf( "thread_range_weighted_sub(): tid %d: sta end = %3d %3d\n",
				        (int)my_id, (int)(*j_start_thr), (int)(*j_end_thr) );
				//printf( "thread_range_weighted_sub(): tid %d: n_left = %3d\n",
				//        (int)my_id, (int)(n) );
				#endif

				// Compute the area of the thread's current subpartition in case
				// the caller is curious how much work they were assigned.
				// NOTE: This area computation isn't actually needed for BLIS to
				// function properly.)
				area = bli_find_area_trap_l( diagoff_j, m, width_j, bf );

				break;
			}

			// Shift the current subpartition's starting and diagonal offsets,
			// as well as the remainder of the n dimension, according to the
			// computed width, and then iterate to the next subpartition.
			off_j     += width_j;
			diagoff_j -= width_j;
			n_left    -= width_j;
		}
	}
	else // if ( bli_is_upper( uplo ) )
	{
		// Express the upper-stored case in terms of the lower-stored case.

		#if 0
		if ( n_way > 1 )
		printf( "thread_range_weighted_sub(): tid %d: m n = %3d %3d do %d (upper)\n",
		        (int)my_id, (int)(m), (int)(n), (int)(diagoff) );
		#endif

		// First, we convert the upper-stored trapezoid to an equivalent
		// lower-stored trapezoid by rotating it 180 degrees.
		bli_rotate180_trapezoid( &diagoff, &uplo, &m, &n );

		// Now that the trapezoid is "flipped" in the n dimension, negate
		// the bool that encodes whether to handle the edge case at the
		// low (or high) end of the index range.
		bli_toggle_bool( &handle_edge_low );

		// Compute the appropriate range for the rotated trapezoid.
		area = bli_thread_range_weighted_sub
		(
		  thread, diagoff, uplo, uplo_orig, m, n, bf,
		  handle_edge_low,
		  j_start_thr, j_end_thr
		);

		// Reverse the indexing basis for the subpartition ranges so that
		// the indices, relative to left-to-right iteration through the
		// unrotated upper-stored trapezoid, map to the correct columns
		// (relative to the diagonal). This amounts to subtracting the
		// range from n.
		bli_reverse_index_direction( n, j_start_thr, j_end_thr );
	}

	return area;
}

// -----------------------------------------------------------------------------

siz_t bli_thread_range_mdim
     (
             dir_t      direct,
             dim_t      bmult,
             bool       use_weighted,
       const thrinfo_t* thr,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
             dim_t*     start,
             dim_t*     end
     )
{
	( void )b;
	return bli_thread_range
	(
	  thr,
	  bli_obj_is_upper_or_lower( c ) ? c : a,
	  bmult,
	  direct,
	  BLIS_M,
	  use_weighted,
	  start,
	  end
	);
}

siz_t bli_thread_range_ndim
     (
             dir_t      direct,
             dim_t      bmult,
             bool       use_weighted,
       const thrinfo_t* thr,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
             dim_t*     start,
             dim_t*     end
     )
{
	( void )a;
	return bli_thread_range
	(
	  thr,
	  bli_obj_is_upper_or_lower( c ) ? c : b,
	  bmult,
	  direct,
	  BLIS_N,
	  use_weighted,
	  start,
	  end
	);
}

// -----------------------------------------------------------------------------

siz_t bli_thread_range
     (
       const thrinfo_t* thr,
       const obj_t*     a,
             dim_t      bf,
             dir_t      direct,
             mdim_t     dim,
             bool       use_weighted,
             dim_t*     start,
             dim_t*     end
     )
{
	dim_t  m       = bli_obj_length( a );
	dim_t  n       = bli_obj_width( a );
	doff_t diagoff = bli_obj_diag_offset( a );
	uplo_t uplo    = bli_obj_uplo( a );

	// Support implicit transposition.
	if ( ( dim == BLIS_M && !bli_obj_has_trans( a ) ) ||
	     ( dim == BLIS_N &&  bli_obj_has_trans( a ) ) )
	{
		bli_reflect_about_diag( &diagoff, &uplo, &m, &n );
	}

	// Edge cases are handled at the "low" end of the index range when
	// moving backwards through the matrix.
	bool handle_edge_low = ( direct == BLIS_BWD );

	if ( use_weighted &&
	     bli_obj_intersects_diag( a ) &&
	     bli_obj_is_upper_or_lower( a ) )
	{
		if ( direct == BLIS_BWD )
		{
			bli_rotate180_trapezoid( &diagoff, &uplo, &m, &n );
		}

		return bli_thread_range_weighted_sub
		(
		  thr, diagoff, uplo, uplo, m, n, bf,
		  handle_edge_low, start, end
		);
	}
	else // if unweighted, dense, or zeros
	{
		bli_thread_range_sub
		(
		  bli_thrinfo_work_id( thr ),
		  bli_thrinfo_n_way( thr ),
		  n,
		  bf,
		  handle_edge_low,
		  start,
		  end
		);

		return m * ( *end - *start );
	}
}

