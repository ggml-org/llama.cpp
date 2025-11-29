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

// -----------------------------------------------------------------------------

#define PRINT_MODE
#define PGUARD if ( 0 )
//#define PRINT_RESULT


#if 0
dim_t bli_thread_range_tlb
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const uplo_t uplo,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	dim_t n_ut_for_me;

	if ( bli_is_lower( uplo ) )
	{
		n_ut_for_me = bli_thread_range_tlb_l
		(
		  nt, tid, diagoff, m_iter, n_iter, mr, nr, j_st_p, i_st_p
		);
	}
	else if ( bli_is_upper( uplo ) )
	{
		n_ut_for_me = bli_thread_range_tlb_u
		(
		  nt, tid, diagoff, m_iter, n_iter, mr, nr, j_st_p, i_st_p
		);
	}
	else // if ( bli_is_dense( uplo ) )
	{
		n_ut_for_me = bli_thread_range_tlb_d
		(
		  nt, tid,          m_iter, n_iter, mr, nr, j_st_p, i_st_p
		);
	}

	return n_ut_for_me;
}
#endif

// -----------------------------------------------------------------------------

dim_t bli_thread_range_tlb_l
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	// This function implements tile-level load balancing for a
	// lower-trapezoidal submatrix. This partitioning guarantees that all
	// threads are assigned nearly the same number of microtiles-worth of work,
	// with a maximum imbalance of one microtile. It makes no effort, however,
	// to account for differences in threads' workload that is attributable to
	// differences in the number of edge-case (or diagonal-intersecting)
	// microtiles (which incur slightly more work since they must first write
	// to a temporary microtile before updating the output C matrix).

	// Assumption: -mr < diagoff. Make sure to prune leading rows beforehand!
	if ( diagoff <= -mr ) bli_abort();

	//
	// -- Step 1: Compute the computational area of the region -----------------
	//

	// Compute the m and n dimensions according to m_iter and n_iter. (These
	// m and n dims will likely be larger than the actual m and n since they
	// "round up" the edge case microtiles into full-sized microtiles.)
	const dim_t m = m_iter * mr;
	const dim_t n = n_iter * nr;

	// For the purposes of many computations in this function, we aren't
	// interested in the extent to which diagoff exceeds n (if it does)
	// So we use a new variable that is guaranteed to be no greater than n.
	const doff_t diagoffmin = bli_min( diagoff, n );

	const dim_t m_rect = m;
	const dim_t n_rect = ( diagoffmin / nr ) * nr;

	const dim_t rect_area    = m_rect * n_rect;
	const dim_t nonrect_area = m * n - rect_area;

	//const dim_t offn_rect       = 0;
	const dim_t offn_nonrect    = n_rect;
	const dim_t diagoff_nonrect = diagoffmin - n_rect; //diagoff % nr;

	const dim_t n_nonrect       = n - n_rect;

	const dim_t offn_ut_nonrect = ( diagoffmin / nr );

	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "min(diagoff,n):     %7ld\n", (long) diagoffmin );
	PGUARD printf( "offn_ut_nonrect:    %7ld\n", (long) offn_ut_nonrect );
	PGUARD printf( "offn_nonrect:       %7ld\n", (long) offn_nonrect );
	PGUARD printf( "diagoff_nonrect:    %7ld\n", (long) diagoff_nonrect );
	PGUARD printf( "n_nonrect:          %7ld\n", (long) n_nonrect );
	PGUARD printf( "---------------------------\n" );

	dim_t num_unref_ut = 0;

	// Count the number of unreferenced utiles strictly above the diagonal.
	for ( dim_t j = 0; j < n_nonrect; j += nr )
	{
		const dim_t diagoff_j = diagoff_nonrect - j;

		// diagoff_j will always be at most nr - 1, but will typically be
		// negative. This is because the non-rectangular region's diagonal
		// offset will be at most nr - 1 for the first column of microtiles,
		// since if it were more than nr - 1, that column would have already
		// been pruned away (via the implicit pruning of diagoff_nonrect).
		// NOTE: We use bli_max() to ensure that -diagoff_j / mr does not
		// become negative, which can only happen if "top" pruning is not
		// performed beforehand (and so it really isn't necessary here).
		const dim_t num_unref_ut_j = bli_max( ( -diagoff_j / mr ), 0 );

		num_unref_ut += num_unref_ut_j;

		PGUARD printf( "j                   %7ld\n", (long) j );
		PGUARD printf( "diagoff_j           %7ld\n", (long) diagoff_j );
		PGUARD printf( "num_unref_ut_j      %7ld\n", (long) num_unref_ut_j );
		PGUARD printf( "num_unref_ut        %7ld\n", (long) num_unref_ut );
		PGUARD printf( "\n" );
	}
	PGUARD printf( "---------------------------\n" );

	const dim_t tri_unref_area = num_unref_ut * mr * nr;
	const dim_t tri_ref_area   = nonrect_area - tri_unref_area;
	const dim_t total_ref_area = rect_area + tri_ref_area;

	PGUARD printf( "gross area:         %7ld\n", (long) ( m * n ) );
	PGUARD printf( "rect_area:          %7ld\n", (long) rect_area );
	PGUARD printf( "nonrect_area:       %7ld\n", (long) nonrect_area );
	PGUARD printf( "tri_unref_area:     %7ld\n", (long) tri_unref_area );
	PGUARD printf( "tri_ref_area:       %7ld\n", (long) tri_ref_area );
	PGUARD printf( "total_ref_area:     %7ld\n", (long) total_ref_area );
	PGUARD printf( "---------------------------\n" );

	//
	// -- Step 2: Compute key utile counts (per thread, per column, etc.) ------
	//

	const dim_t n_ut_ref      = total_ref_area / ( mr * nr );
	//const dim_t n_ut_tri_ref  = tri_ref_area   / ( mr * nr );
	const dim_t n_ut_rect     = rect_area      / ( mr * nr );

	PGUARD printf( "n_ut_ref:           %7ld\n", (long) n_ut_ref );
	//PGUARD printf( "n_ut_tri_ref:       %7ld\n", (long) n_ut_tri_ref );
	PGUARD printf( "n_ut_rect:          %7ld\n", (long) n_ut_rect );
	PGUARD printf( "---------------------------\n" );

	// Compute the number of microtiles to allocate per thread as well as the
	// number of leftover microtiles.
	const dim_t n_ut_per_thr = n_ut_ref / nt;
	const dim_t n_ut_pt_left = n_ut_ref % nt;

	PGUARD printf( "n_ut_per_thr:       %7ld\n", (long) n_ut_per_thr );
	PGUARD printf( "n_ut_pt_left:       %7ld\n", (long) n_ut_pt_left );
	PGUARD printf( "---------------------------\n" );

	const dim_t n_ut_per_col = m_iter;

	PGUARD printf( "n_ut_per_col:       %7ld\n", (long) n_ut_per_col );

	// Allocate one of the leftover microtiles to the current thread if its
	// tid is one of the lower thread ids.
	const dim_t n_ut_for_me = n_ut_per_thr + ( tid < n_ut_pt_left ? 1 : 0 );

	PGUARD printf( "n_ut_for_me:        %7ld (%ld+%ld)\n", (long) n_ut_for_me,
	               (long) n_ut_per_thr, (long) ( n_ut_for_me - n_ut_per_thr ) );

	// Compute the number of utiles prior to the current thread's starting
	// point. This is the sum of all n_ut_for_me for all thread ids less
	// than tid. Notice that the second half of this expression effectively
	// adds one extra microtile for each lower-valued thread id, up to
	// n_ut_pt_left.
	const dim_t n_ut_before = tid * n_ut_per_thr + bli_min( tid, n_ut_pt_left );

	PGUARD printf( "n_ut_before:        %7ld\n", (long) n_ut_before );
	PGUARD printf( "---------------------------\n" );

	//
	// -- Step 3: Compute the starting j/i utile offset for a given tid --------
	//

	dim_t j_st;
	dim_t i_st;

	if ( n_ut_before < n_ut_rect )
	{
		// This branch handles scenarios where the number of microtiles
		// assigned to lower thread ids is strictly less than the number of
		// utiles in the rectangular region. This means that calculating the
		// starting microtile index is easy (because it does not need to
		// take the location of the diagonal into account).

		PGUARD printf( "Rectangular region: n_ut_before < n_ut_rect\n" );
		PGUARD printf( "\n" );

		const dim_t ut_index_rect_st = n_ut_before;

		PGUARD printf( "ut_index_st:        %7ld\n", (long) ut_index_rect_st );
		PGUARD printf( "---------------------------\n" );

		j_st = ut_index_rect_st / n_ut_per_col;
		i_st = ut_index_rect_st % n_ut_per_col;

		PGUARD printf( "j_st, i_st (fnl=)      %4ld,%4ld\n",
		               (long) j_st, (long) i_st );
	}
	else // if ( n_ut_rect <= n_ut_before )
	{
		// This branch handles scenarios where the number of microtiles
		// assigned to lower thread ids exceeds (or equals) the number of
		// utiles in the rectangular region. This means we need to observe the
		// location of the diagonal to see how many utiles are referenced per
		// column of utiles.

		PGUARD printf( "Diagonal region: n_ut_rect <= n_ut_before\n" );
		PGUARD printf( "\n" );

		// This will be the number of microtile columns we will immediately
		// advance past to get to the diagonal region.
		const dim_t n_ut_col_adv = offn_ut_nonrect;

		PGUARD printf( "n_ut_col_adv:       %7ld\n", (long) n_ut_col_adv );

		// In order to find j_st and i_st, we need to "allocate" n_ut_before
		// microtiles.
		dim_t n_ut_tba = n_ut_before;

		PGUARD printf( "n_ut_tba:           %7ld\n", (long) n_ut_tba );

		// Advance past the rectangular region, decrementing n_ut_tba
		// accordingly.
		n_ut_tba -= n_ut_per_col * n_ut_col_adv;

		PGUARD printf( "n_ut_tba_1:         %7ld\n", (long) n_ut_tba );
		PGUARD printf( "\n" );

		// In case n_ut_tba == 0. Only happens when n_ut_before == n_ut_rect.
		j_st = n_ut_col_adv;
		i_st = 0;

		for ( dim_t j = n_ut_col_adv; 0 < n_ut_tba; ++j )
		{
			const dim_t diagoff_j     = diagoffmin - j*nr;
			const dim_t n_ut_skip_j   = bli_max( -diagoff_j / mr, 0 );
			const dim_t n_ut_this_col = n_ut_per_col - n_ut_skip_j;

			PGUARD printf( "j:                  %7ld\n", (long) j );
			PGUARD printf( "diagoff_j:          %7ld\n", (long) diagoff_j );
			PGUARD printf( "n_ut_skip_j:        %7ld\n", (long) n_ut_skip_j );
			PGUARD printf( "n_ut_this_col:      %7ld\n", (long) n_ut_this_col );
			PGUARD printf( "n_ut_tba_j0:        %7ld\n", (long) n_ut_tba );

			if ( n_ut_tba < n_ut_this_col )
			{
				// If the number of utiles to allocate is less than the number
				// in this column, we know that j_st will refer to the current
				// column. To find i_st, we first skip to the utile that
				// intersects the diagonal and then add n_ut_tba.
				j_st = j;
				i_st = n_ut_skip_j + n_ut_tba;
				PGUARD printf( "j_st, i_st (fnl<)      %4ld,%4ld\n",
				               (long)  j_st, (long) i_st );
			}
			else if ( n_ut_tba == n_ut_this_col )
			{
				// If the number of utiles to allocate is exactly equal to the
				// number in this column, we know that j_st will refer to the
				// *next* column. But to find i_st, we will have to take the
				// location of the diagonal into account.
				const doff_t diagoff_jp1   = diagoff_j - nr;
				const dim_t  n_ut_skip_jp1 = bli_max( -diagoff_jp1 / mr, 0 );

				j_st = j + 1;
				i_st = n_ut_skip_jp1;
				PGUARD printf( "j_st, i_st (fnl=)      %4ld,%4ld\n",
				               (long) j_st, (long) i_st );
			}

			// No matter what (especially if the number of utiles to allocate
			// exceeds the number in this column), we decrement n_ut_tba attempt
			// to continue to the next iteration. (Note: If either of the two
			// branches above is triggered, n_ut_tba will be decremented down to
			// zero (or less), in which case this will be the final iteration.)
			n_ut_tba -= n_ut_this_col;

			PGUARD printf( "n_ut_tba_j1:        %7ld\n", (long) n_ut_tba );
			PGUARD printf( "\n" );
		}
	}

	//
	// -- Step 4: Save the results ---------------------------------------------
	//

	*j_st_p = j_st;
	*i_st_p = i_st;

	#ifdef PRINT_RESULT
	printf( "j_st, i_st (mem)       %4ld,%4ld  (n_ut: %4ld)\n",
	        (long) j_st, (long) i_st, (long) n_ut_for_me );
	#endif

	// Return the number of utiles that this thread was allocated.
	return n_ut_for_me;
}

// -----------------------------------------------------------------------------

dim_t bli_thread_range_tlb_u
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	// This function implements tile-level load balancing for an
	// upper-trapezoidal submatrix. This partitioning guarantees that all
	// threads are assigned nearly the same number of microtiles-worth of work,
	// with a maximum imbalance of one microtile. It makes no effort, however,
	// to account for differences in threads' workload that is attributable to
	// differences in the number of edge-case (or diagonal-intersecting)
	// microtiles (which incur slightly more work since they must first write
	// to a temporary microtile before updating the output C matrix).

	// Assumption: diagoff < nr. Make sure to prune leading columns beforehand!
	if ( nr <= diagoff ) bli_abort();

	//
	// -- Step 1: Compute the computational area of the region -----------------
	//

	// Compute the m and n dimensions according to m_iter and n_iter. (These
	// m and n dims will likely be larger than the actual m and n since they
	// "round up" the edge case microtiles into full-sized microtiles.)
	const dim_t m = m_iter * mr;
	const dim_t n = n_iter * nr;

	// For the purposes of many computations in this function, we aren't
	// interested in the extent to which diagoff exceeds -m (if it does)
	// So we use a new variable that is guaranteed to be no less than -m.
	const doff_t diagoffmin = bli_max( diagoff, -m );

	const dim_t m_rect = m;
	const dim_t n_rect = ( -diagoffmin / nr ) * nr;

	const dim_t rect_area    = m_rect * n_rect;
	const dim_t nonrect_area = m * n - rect_area;

	const dim_t offn_rect       = n - n_rect;
	//const dim_t offn_nonrect    = 0;
	const dim_t diagoff_nonrect = diagoffmin;

	const dim_t n_nonrect       = n - n_rect;

	const dim_t offn_ut_rect    = n_iter + ( diagoffmin / nr );

	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "max(diagoff,-m):    %7ld\n", (long) diagoffmin );
	PGUARD printf( "offn_ut_rect:       %7ld\n", (long) offn_ut_rect );
	PGUARD printf( "offn_rect:          %7ld\n", (long) offn_rect );
	PGUARD printf( "diagoff_nonrect:    %7ld\n", (long) diagoff_nonrect );
	PGUARD printf( "n_nonrect:          %7ld\n", (long) n_nonrect );
	PGUARD printf( "---------------------------\n" );

	dim_t num_unref_ut = 0;

	// Count the number of unreferenced utiles strictly below the diagonal.
	for ( dim_t j = 0; j < n_nonrect; j += nr )
	{
		const dim_t diagoff_j = diagoff_nonrect - j;

		// diagoff_j will always be at most nr - 1, but will typically be
		// negative. This is because the non-rectangular region's diagonal
		// offset will be at most nr - 1 for the first column of microtiles,
		// since if it were more than nr - 1, that column would have already
		// been pruned away (prior to this function being called).
		// NOTE: We use bli_max() to ensure that ( m + diagoff_j - nr ) / mr
		// does not become negative, which can happen in some situations
		// during the first iteration if diagoff is relatively close to -m.
		// NOTE: We subtract nr from diagoff_j since it's really the diagonal
		// offset of the *next* column of utiles that needs to be used to
		// determine how many utiles are referenced in the current column.
		const dim_t num_unref_ut_j = bli_max( ( m + diagoff_j - nr ) / mr, 0 );

		num_unref_ut += num_unref_ut_j;

		PGUARD printf( "j                   %7ld\n", (long) j );
		PGUARD printf( "diagoff_j - nr      %7ld\n", (long) ( diagoff_j - nr ) );
		PGUARD printf( "num_unref_ut_j      %7ld\n", (long) num_unref_ut_j );
		PGUARD printf( "num_unref_ut        %7ld\n", (long) num_unref_ut );
		PGUARD printf( "\n" );
	}
	PGUARD printf( "---------------------------\n" );

	const dim_t tri_unref_area = num_unref_ut * mr * nr;
	const dim_t tri_ref_area   = nonrect_area - tri_unref_area;
	const dim_t total_ref_area = rect_area + tri_ref_area;

	PGUARD printf( "gross area:         %7ld\n", (long) ( m * n ) );
	PGUARD printf( "rect_area:          %7ld\n", (long) rect_area );
	PGUARD printf( "nonrect_area:       %7ld\n", (long) nonrect_area );
	PGUARD printf( "tri_unref_area:     %7ld\n", (long) tri_unref_area );
	PGUARD printf( "tri_ref_area:       %7ld\n", (long) tri_ref_area );
	PGUARD printf( "total_ref_area:     %7ld\n", (long) total_ref_area );
	PGUARD printf( "---------------------------\n" );

	//
	// -- Step 2: Compute key utile counts (per thread, per column, etc.) ------
	//

	const dim_t n_ut_ref      = total_ref_area / ( mr * nr );
	const dim_t n_ut_tri_ref  = tri_ref_area   / ( mr * nr );
	//const dim_t n_ut_rect     = rect_area      / ( mr * nr );

	PGUARD printf( "n_ut_ref:           %7ld\n", (long) n_ut_ref );
	PGUARD printf( "n_ut_tri_ref:       %7ld\n", (long) n_ut_tri_ref );
	//PGUARD printf( "n_ut_rect:          %7ld\n", n_ut_rect );
	PGUARD printf( "---------------------------\n" );

	// Compute the number of microtiles to allocate per thread as well as the
	// number of leftover microtiles.
	const dim_t n_ut_per_thr = n_ut_ref / nt;
	const dim_t n_ut_pt_left = n_ut_ref % nt;

	PGUARD printf( "n_ut_per_thr:       %7ld\n", (long) n_ut_per_thr );
	PGUARD printf( "n_ut_pt_left:       %7ld\n", (long) n_ut_pt_left );
	PGUARD printf( "---------------------------\n" );

	const dim_t n_ut_per_col = m_iter;

	PGUARD printf( "n_ut_per_col:       %7ld\n", (long) n_ut_per_col );

	// Allocate one of the leftover microtiles to the current thread if its
	// tid is one of the lower thread ids.
	const dim_t n_ut_for_me = n_ut_per_thr + ( tid < n_ut_pt_left ? 1 : 0 );

	PGUARD printf( "n_ut_for_me:        %7ld (%ld+%ld)\n", (long) n_ut_for_me,
	               (long) n_ut_per_thr, (long) ( n_ut_for_me - n_ut_per_thr ) );

	// Compute the number of utiles prior to the current thread's starting
	// point. This is the sum of all n_ut_for_me for all thread ids less
	// than tid. Notice that the second half of this expression effectively
	// adds one extra microtile for each lower-valued thread id, up to
	// n_ut_pt_left.
	const dim_t n_ut_before = tid * n_ut_per_thr + bli_min( tid, n_ut_pt_left );

	PGUARD printf( "n_ut_before:        %7ld\n", (long) n_ut_before );
	PGUARD printf( "---------------------------\n" );

	//
	// -- Step 3: Compute the starting j/i utile offset for a given tid --------
	//

	dim_t j_st;
	dim_t i_st;

	if ( n_ut_tri_ref <= n_ut_before )
	{
		// This branch handles scenarios where the number of microtiles
		// assigned to lower thread ids exceeds (or equals) the number of
		// utiles in the diagonal region. This means that calculating the
		// starting microtile index is easy (because it does not need to
		// take the location of the diagonal into account).

		PGUARD printf( "Rectangular region: n_ut_tri_ref <= n_ut_before\n" );
		PGUARD printf( "\n" );

		const dim_t ut_index_rect_st = n_ut_before - n_ut_tri_ref;

		PGUARD printf( "ut_index_rect_st:   %7ld\n", (long) ut_index_rect_st );
		PGUARD printf( "---------------------------\n" );

		j_st = offn_ut_rect + ut_index_rect_st / n_ut_per_col;
		i_st =                ut_index_rect_st % n_ut_per_col;

		PGUARD printf( "j_st, i_st (fnl=)      %4ld,%4ld\n",
		               (long) j_st, (long) i_st );
	}
	else // if ( n_ut_before < n_ut_tri_ref )
	{
		// This branch handles scenarios where the number of microtiles
		// assigned to lower thread ids is strictly less than the number of
		// utiles in the diagonal region. This means we need to observe the
		// location of the diagonal to see how many utiles are referenced per
		// column of utiles.

		PGUARD printf( "Diagonal region: n_ut_before < n_ut_tri_ref\n" );
		PGUARD printf( "\n" );

		// This will be the number of microtile columns we will immediately
		// advance past to get to the diagonal region.
		const dim_t n_ut_col_adv = 0;

		PGUARD printf( "n_ut_col_adv:       %7ld\n", (long) n_ut_col_adv );

		// In order to find j_st and i_st, we need to "allocate" n_ut_before
		// microtiles.
		dim_t n_ut_tba = n_ut_before;

		PGUARD printf( "n_ut_tba:           %7ld\n", (long) n_ut_tba );

		// No need to advance since the upper-trapezoid begins with the
		// diagonal region.
		//n_ut_tba -= 0;

		PGUARD printf( "n_ut_tba_1:         %7ld\n", (long) n_ut_tba );
		PGUARD printf( "\n" );

		// In case n_ut_tba == 0. Only happens when n_ut_before == 0.
		j_st = 0;
		i_st = 0;

		for ( dim_t j = n_ut_col_adv; 0 < n_ut_tba; ++j )
		{
			const dim_t diagoff_j     = diagoffmin - j*nr;
			const dim_t n_ut_skip_j   = bli_max( ( m + diagoff_j - nr ) / mr, 0 );
			const dim_t n_ut_this_col = n_ut_per_col - n_ut_skip_j;

			PGUARD printf( "j:                  %7ld\n", (long) j );
			PGUARD printf( "diagoff_j:          %7ld\n", (long) diagoff_j );
			PGUARD printf( "n_ut_skip_j:        %7ld\n", (long) n_ut_skip_j );
			PGUARD printf( "n_ut_this_col:      %7ld\n", (long) n_ut_this_col );
			PGUARD printf( "n_ut_tba_j0:        %7ld\n", (long) n_ut_tba );

			if ( n_ut_tba < n_ut_this_col )
			{
				// If the number of utiles to allocate is less than the number
				// in this column, we know that j_st will refer to the current
				// column. To find i_st, we simply use n_ut_tba.
				j_st = j;
				i_st = n_ut_tba;
				PGUARD printf( "j_st, i_st (fnl<)      %4ld,%4ld\n",
				               (long) j_st, (long) i_st );
			}
			else if ( n_ut_tba == n_ut_this_col )
			{
				// If the number of utiles to allocate is exactly equal to the
				// number in this column, we know that j_st will refer to the
				// *next* column. In this situation, i_st will always be 0.

				j_st = j + 1;
				i_st = 0;
				PGUARD printf( "j_st, i_st (fnl=)      %4ld,%4ld\n",
				               (long) j_st, (long) i_st );
			}

			// No matter what (especially if the number of utiles to allocate
			// exceeds the number in this column), we decrement n_ut_tba attempt
			// to continue to the next iteration. (Note: If either of the two
			// branches above is triggered, n_ut_tba will be decremented down to
			// zero (or less), in which case this will be the final iteration.)
			n_ut_tba -= n_ut_this_col;

			PGUARD printf( "n_ut_tba_j1:        %7ld\n", (long) n_ut_tba );
			PGUARD printf( "\n" );
		}
	}

	//
	// -- Step 4: Save the results ---------------------------------------------
	//

	*j_st_p = j_st;
	*i_st_p = i_st;

	#ifdef PRINT_RESULT
	printf( "j_st, i_st (mem)       %4ld,%4ld  (n_ut: %4ld)\n",
	        (long) j_st, (long) i_st, (long) n_ut_for_me );
	#endif

	// Return the number of utiles that this thread was allocated.
	return n_ut_for_me;
}

// -----------------------------------------------------------------------------

dim_t bli_thread_range_tlb_d
     (
       const dim_t  nt,
       const dim_t  tid,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	// This function implements tile-level load balancing for a
	// general/dense submatrix. This partitioning guarantees that all
	// threads are assigned nearly the same number of microtiles-worth of work,
	// with a maximum imbalance of one microtile. It makes no effort, however,
	// to account for differences in threads' workload that is attributable to
	// differences in the number of edge-case microtiles (which incur slightly
	// more work since they must first write to a temporary microtile before
	// updating the output C matrix).

	//
	// -- Step 1: Compute the computational area of the region -----------------
	//

	// Compute the m and n dimensions according to m_iter and n_iter. (These
	// m and n dims will likely be larger than the actual m and n since they
	// "round up" the edge case microtiles into full-sized microtiles.)
	const dim_t m = m_iter * mr;
	const dim_t n = n_iter * nr;

	const dim_t m_rect = m;
	const dim_t n_rect = n;

	const dim_t total_ref_area = m_rect * n_rect;

	PGUARD printf( "total_ref_area:     %7ld\n", (long) total_ref_area );
	PGUARD printf( "---------------------------\n" );

	//
	// -- Step 2: Compute key utile counts (per thread, per column, etc.) ------
	//

	const dim_t n_ut_ref = total_ref_area / ( mr * nr );

	PGUARD printf( "n_ut_ref:           %7ld\n", (long) n_ut_ref );
	PGUARD printf( "---------------------------\n" );

	// Compute the number of microtiles to allocate per thread as well as the
	// number of leftover microtiles.
	const dim_t n_ut_per_thr = n_ut_ref / nt;
	const dim_t n_ut_pt_left = n_ut_ref % nt;

	PGUARD printf( "n_ut_per_thr:       %7ld\n", (long) n_ut_per_thr );
	PGUARD printf( "n_ut_pt_left:       %7ld\n", (long) n_ut_pt_left );
	PGUARD printf( "---------------------------\n" );

	const dim_t n_ut_per_col = m_iter;

	PGUARD printf( "n_ut_per_col:       %7ld\n", (long) n_ut_per_col );

	// Allocate one of the leftover microtiles to the current thread if its
	// tid is one of the lower thread ids.
	const dim_t n_ut_for_me = n_ut_per_thr + ( tid < n_ut_pt_left ? 1 : 0 );

	PGUARD printf( "n_ut_for_me:        %7ld (%ld+%ld)\n", (long) n_ut_for_me,
	               (long) n_ut_per_thr, (long) ( n_ut_for_me - n_ut_per_thr ) );

	// Compute the number of utiles prior to the current thread's starting
	// point. This is the sum of all n_ut_for_me for all thread ids less
	// than tid. Notice that the second half of this expression effectively
	// adds one extra microtile for each lower-valued thread id, up to
	// n_ut_pt_left.
	const dim_t n_ut_before = tid * n_ut_per_thr + bli_min( tid, n_ut_pt_left );

	PGUARD printf( "n_ut_before:        %7ld\n", (long) n_ut_before );
	PGUARD printf( "---------------------------\n" );

	//
	// -- Step 3: Compute the starting j/i utile offset for a given tid --------
	//

	const dim_t ut_index_st = n_ut_before;

	PGUARD printf( "ut_index_st:        %7ld\n", (long) ut_index_st );
	PGUARD printf( "---------------------------\n" );

	const dim_t j_st = ut_index_st / n_ut_per_col;
	const dim_t i_st = ut_index_st % n_ut_per_col;

	//
	// -- Step 4: Save the results ---------------------------------------------
	//

	*j_st_p = j_st;
	*i_st_p = i_st;

	#ifdef PRINT_RESULT
	printf( "j_st, i_st (mem)       %4ld,%4ld  (n_ut: %4ld)\n",
	        (long) j_st, (long) i_st, (long) n_ut_for_me );
	#endif

	// Return the number of utiles that this thread was allocated.
	return n_ut_for_me;
}

// -----------------------------------------------------------------------------

BLIS_INLINE dim_t bli_tlb_trmm_lx_k_iter
     (
       const doff_t diagoff_iter,
       const uplo_t uplo,
       const dim_t  k_iter,
       const dim_t  ir_iter
     )
{
	if ( bli_is_lower( uplo ) )
		return bli_min( diagoff_iter + ( ir_iter + 1 ), k_iter );
	else // if ( bli_is_upper( uplo ) )
		return k_iter - bli_max( diagoff_iter + ir_iter, 0 );
}

BLIS_INLINE dim_t bli_tlb_trmm_rl_k_iter
     (
       const doff_t diagoff_iter,
       const dim_t  k_iter,
       const dim_t  jr_iter
     )
{
	return k_iter - bli_max( -diagoff_iter + jr_iter, 0 );
}

// -----------------------------------------------------------------------------

dim_t bli_thread_range_tlb_trmm_ll
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	return bli_thread_range_tlb_trmm_lx_impl
	(
	  nt, tid, diagoff, BLIS_LOWER, m_iter, n_iter, k_iter, mr, nr,
	  j_st_p, i_st_p
	);
}

dim_t bli_thread_range_tlb_trmm_lu
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	return bli_thread_range_tlb_trmm_lx_impl
	(
	  nt, tid, diagoff, BLIS_UPPER, m_iter, n_iter, k_iter, mr, nr,
	  j_st_p, i_st_p
	);
}

dim_t bli_thread_range_tlb_trmm_lx_impl
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const uplo_t uplo,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	// Assumption: 0 <= diagoff (lower); diagoff <= 0 (upper).
	// Make sure to prune leading rows (lower) or columns (upper) beforehand!
	if      ( bli_is_lower( uplo ) && diagoff < 0 ) bli_abort();
	else if ( bli_is_upper( uplo ) && diagoff > 0 ) bli_abort();

	// Single-threaded cases are simple and allow early returns.
	if ( nt == 1 )
	{
		const dim_t n_ut_for_me = m_iter * n_iter;

		*j_st_p = 0;
		*i_st_p = 0;

		return n_ut_for_me;
	}

	//
	// -- Step 1: Compute the computational flop cost of each utile column -----
	//

	// Normalize the diagonal offset by mr so that it represents the offset in
	// units of mr x mr chunks.
	const doff_t diagoff_iter = diagoff / mr;

	// Determine the actual k dimension, in units of mr x mr iterations, capped
	// by the k_iter given by the caller.

	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "m_iter:             %7ld\n", (long) m_iter );
	PGUARD printf( "n_iter:             %7ld\n", (long) n_iter );
	PGUARD printf( "k_iter:             %7ld\n", (long) k_iter );
	PGUARD printf( "mr:                 %7ld\n", (long) mr );
	PGUARD printf( "nr:                 %7ld\n", (long) nr );
	PGUARD printf( "diagoff_iter:       %7ld\n", (long) diagoff_iter );

	dim_t uops_per_col = 0;

	// Compute the computation flop cost of each microtile column, normalized
	// by the number of flops performed by each mr x nr rank-1 update. This
	// is simply the sum of all of the k dimensions of each micropanel, up to
	// and including (lower) or starting from (upper) the part that intersects
	// the diagonal, or the right (lower) or left (upper) edge of the matrix,
	// as applicable.
	for ( dim_t i = 0; i < m_iter; ++i )
	{
		// Don't allow k_a1011 to exceed k_iter, which is the maximum possible
		// k dimension (in units of mr x mr chunks of micropanel).
		const dim_t k_i_iter
		= bli_tlb_trmm_lx_k_iter( diagoff_iter, uplo, k_iter, i );

		uops_per_col += k_i_iter;
	}

	PGUARD printf( "uops_per_col:       %7ld\n", (long) uops_per_col );

	//
	// -- Step 2: Compute key flop counts (per thread, per column, etc.) -------
	//

	// Compute the total cost for the entire block-panel multiply.
	const dim_t total_uops = uops_per_col * n_iter;

	// Compute the number of microtile ops to allocate per thread as well as the
	// number of leftover microtile ops.
	const dim_t n_uops_per_thr = total_uops / nt;
	const dim_t n_uops_pt_left = total_uops % nt;

	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "total_uops:         %7ld\n", (long) total_uops );
	PGUARD printf( "n_uops_per_thr:     %7ld\n", (long) n_uops_per_thr );
	PGUARD printf( "n_uops_pt_left:     %7ld\n", (long) n_uops_pt_left );

	//
	// -- Step 3: Compute the starting j/i utile offset for a given tid --------
	//

	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "total_utiles:       %7ld\n", (long) ( m_iter * n_iter ) );
	PGUARD printf( "---------------------------\n" );

	dim_t j_st_cur = 0; dim_t j_en_cur = 0;
	dim_t i_st_cur = 0; dim_t i_en_cur = 0;

	PGUARD printf( "          tid %ld will start at j,i: %ld %ld\n",
	               (long) 0, (long) j_st_cur, (long) i_st_cur );

	// Find the utile update that pushes uops_tba to 0 or less.
#ifdef PRINT_MODE
	for ( dim_t tid_i = 0; tid_i < nt; ++tid_i )
#else
	for ( dim_t tid_i = 0; tid_i < nt - 1; ++tid_i )
#endif
	{
		const dim_t uops_ta     = n_uops_per_thr + ( tid_i < n_uops_pt_left ? 1 : 0 );
		      dim_t uops_tba    = uops_ta;
		      dim_t j           = j_st_cur;
		      dim_t n_ut_for_me = 0;
		      bool  done_e      = FALSE;

		PGUARD printf( "tid_i: %ld  n_uops to alloc: %3ld \n", (long) tid_i, (long) uops_tba );

		// This code begins allocating uops when the starting point is somewhere
		// after the first microtile. Typically this will not be enough to
		// allocate all uops, except for small matrices (and/or high numbers of
		// threads), in which case the code signals an early finish (via done_e).
		if ( 0 < i_st_cur )
		{
			dim_t i;

			//PGUARD printf( "tid_i: %ld  uops left to alloc: %2ld \n", (long) tid_i, (long) uops_tba );

			for ( i = i_st_cur; i < m_iter; ++i )
			{
				n_ut_for_me += 1;

				const dim_t uops_tba_new
				= uops_tba -
				  bli_tlb_trmm_lx_k_iter( diagoff_iter, uplo, k_iter, i );

				uops_tba = uops_tba_new;

				PGUARD printf( "tid_i: %ld  i: %2ld  (1 n_ut_cur: %ld) (uops_alloc: %ld)\n",
				               (long) tid_i, (long) i, (long) n_ut_for_me,
				               (long) ( uops_ta - uops_tba ) );

				if ( uops_tba_new <= 0 ) { j_en_cur = j; i_en_cur = i; done_e = TRUE;
				                           break; }
			}

			if ( i == m_iter ) j += 1;
		}

		// This code advances over as many columns of utiles as possible and then
		// walks down to the correct utile within the subsequent column. However,
		// it gets skipped entirely if the previous code block was able to
		// allocate all of the current tid's uops.
		if ( !done_e )
		{
			const dim_t j_inc0  = uops_tba / uops_per_col;
			const dim_t j_left0 = uops_tba % uops_per_col;

			// We need to set a hard limit on how much j_inc can be. Namely,
			// it should not exceed the number of utile columns that are left
			// in the matrix. We also correctly compute j_left when the initial
			// computation of j_inc0 above exceeds the revised j_inc, but this
			// is mostly only so that in these situations the debug statements
			// report the correct numbers.
			const dim_t j_inc  = bli_min( j_inc0, n_iter - j );
			const dim_t delta  = j_inc0 - j_inc;
			const dim_t j_left = j_left0 + delta * uops_per_col;

			// Increment j by the number of full utile columns we allocate, and
			// set the remaining utile ops to be allocated to the remainder.
			j       += j_inc;
			uops_tba = j_left;

			n_ut_for_me += j_inc * m_iter;

			PGUARD printf( "tid_i: %ld  advanced to col: %2ld  (uops traversed: %ld)\n",
			               (long) tid_i, (long) j, (long) ( uops_per_col * j_inc ) );
			PGUARD printf( "tid_i: %ld  j: %2ld  (  n_ut_cur: %ld) (uops_alloc: %ld)\n",
			               (long) tid_i, (long) j, (long) n_ut_for_me,
			               (long) ( uops_ta - uops_tba ) );
			PGUARD printf( "tid_i: %ld  uops left to alloc: %2ld \n",
			               (long) tid_i, (long) j_left );

			if ( uops_tba == 0 )
			{
				// If advancing j_inc columns allocated all of our uops, then
				// designate the last iteration of the previous column as the
				// end point.
				j_en_cur = j - 1;
				i_en_cur = m_iter - 1;
			}
			else if ( j >  n_iter ) bli_abort(); // safety check.
			else if ( j == n_iter )
			{
				// If we still have at least some uops to allocate, and advancing
				// j_inc columns landed us at the beginning of the first non-
				// existent column (column n_iter), then we're done. (The fact
				// that we didn't get to allocate all of our uops just means that
				// the lower tids slightly overshot their allocations, leaving
				// fewer uops for the last thread.)
			}
			else // if ( 0 < uops_tba && j < n_iter )
			{
				// If we have at least some uops to allocate, and we still have
				// at least some columns to process, then we search for the
				// utile that will put us over the top.

				for ( dim_t i = 0; i < m_iter; ++i )
				{
					n_ut_for_me += 1;

					const dim_t uops_tba_new
					= uops_tba -
					  bli_tlb_trmm_lx_k_iter( diagoff_iter, uplo, k_iter, i );

					uops_tba = uops_tba_new;

					PGUARD printf( "tid_i: %ld  i: %2ld  (4 n_ut_cur: %ld) (uops_alloc: %ld)\n",
					               (long) tid_i, (long) i,
					               (long) n_ut_for_me, (long) ( uops_ta - uops_tba ) );

					if ( uops_tba_new <= 0 ) { j_en_cur = j; i_en_cur = i;
					                           break; }
				}
			}
		}


		PGUARD printf( "tid_i: %ld         (5 n_ut_cur: %ld) (overshoot: %ld out of %ld)\n",
		               (long)  tid_i, (long) n_ut_for_me, -(long) uops_tba, (long) uops_ta );

		if ( tid_i == tid )
		{
			*j_st_p = j_st_cur;
			*i_st_p = i_st_cur;
			return n_ut_for_me;
		}

		// Use the current tid's ending i,j values to determine the starting i,j
		// values for the next tid.
		j_st_cur = j_en_cur;
		i_st_cur = i_en_cur + 1;
		if ( i_st_cur == m_iter ) { j_st_cur += 1; i_st_cur = 0; }

		PGUARD printf( "tid_i: %ld         (6 n_ut_cur: %ld)\n",
		               (long) tid_i, (long) n_ut_for_me );
		PGUARD printf( "tid_i: %ld  tid %ld will start at j,i: %ld %ld\n",
		               (long) tid_i, (long) tid_i + 1,
		               (long) j_st_cur, (long) i_st_cur );
		PGUARD printf( "---------------------------\n" );
	}

#ifndef PRINT_MODE

	//
	// -- Step 4: Handle the last thread's allocation --------------------------
	//

	// An optimization: The above loop runs to nt - 1 rather than nt since it's
	// easy to count the number of utiles allocated to the last thread.
	const dim_t n_ut_for_me = m_iter - i_st_cur +
	                          (n_iter - j_st_cur - 1) * m_iter;
	*j_st_p = j_st_cur;
	*i_st_p = i_st_cur;

	PGUARD printf( "tid_i: %ld         (7 n_ut_for_me: %ld) (j,i_st: %ld %ld)\n",
	               (long) tid, (long) n_ut_for_me,
	               (long) j_st_cur, (long) i_st_cur );

	return n_ut_for_me;
#else
	// This line should never execute, but we need it to satisfy the compiler.
	return -1;
#endif
}

// -----------------------------------------------------------------------------

#if 0
dim_t bli_thread_range_tlb_trmm_r
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const uplo_t uplo,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	dim_t n_ut_for_me;

	if ( bli_is_lower( uplo ) )
	{
		inc_t j_en_l, i_en_l;

		n_ut_for_me = bli_thread_range_tlb_trmm_rl_impl
		(
		  nt, tid, diagoff, m_iter, n_iter, k_iter, mr, nr,
		  j_st_p, i_st_p, &j_en_l, &i_en_l
		);
	}
	else // if ( bli_is_upper( uplo ) )
	{
		inc_t j_st_l, i_st_l;
		inc_t j_en_l, i_en_l;

		// Reverse the effective tid and use the diagonal offset as if the m and
		// n dimension were reversed (similar to a 180 degree rotation). This
		// transforms the problem into one of allocating ranges for a lower-
		// triangular matrix, for which we already have a special routine.
		const dim_t  tid_rev     = nt - tid - 1;
		const doff_t diagoff_rev = nr*n_iter - ( nr*k_iter + diagoff );

		n_ut_for_me = bli_thread_range_tlb_trmm_rl_impl
		(
		  nt, tid_rev, diagoff_rev, m_iter, n_iter, k_iter, mr, nr,
		  &j_st_l, &i_st_l, &j_en_l, &i_en_l
		);

		// The ending j and i offsets will serve as our starting offsets
		// returned to the caller, but first we have to reverse the offsets so
		// that their semantics are once again relative to an upper-triangular
		// matrix.
		j_en_l = n_iter - j_en_l - 1;
		i_en_l = m_iter - i_en_l - 1;

		*j_st_p = j_en_l;
		*i_st_p = i_en_l;
	}

	return n_ut_for_me;
}
#endif

dim_t bli_thread_range_tlb_trmm_rl
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	inc_t j_en_l, i_en_l;

	return bli_thread_range_tlb_trmm_rl_impl
	(
	  nt, tid, diagoff, m_iter, n_iter, k_iter, mr, nr,
	  j_st_p, i_st_p, &j_en_l, &i_en_l
	);
}

dim_t bli_thread_range_tlb_trmm_ru
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p
     )
{
	inc_t j_st_l, i_st_l;
	inc_t j_en_l, i_en_l;

	// Reverse the effective tid and use the diagonal offset as if the m and
	// n dimension were reversed (similar to a 180 degree rotation). This
	// transforms the problem into one of allocating ranges for a lower-
	// triangular matrix, for which we already have a special routine.
	const dim_t  tid_rev     = nt - tid - 1;
	const doff_t diagoff_rev = nr*n_iter - ( nr*k_iter + diagoff );

	const dim_t n_ut_for_me = bli_thread_range_tlb_trmm_rl_impl
	(
	  nt, tid_rev, diagoff_rev, m_iter, n_iter, k_iter, mr, nr,
	  &j_st_l, &i_st_l, &j_en_l, &i_en_l
	);

	// The ending j and i offsets will serve as our starting offsets
	// returned to the caller, but first we have to reverse the offsets so
	// that their semantics are once again relative to an upper-triangular
	// matrix.
	j_en_l = n_iter - j_en_l - 1;
	i_en_l = m_iter - i_en_l - 1;

	*j_st_p = j_en_l;
	*i_st_p = i_en_l;

	return n_ut_for_me;
}

dim_t bli_thread_range_tlb_trmm_rl_impl
     (
       const dim_t  nt,
       const dim_t  tid,
       const doff_t diagoff,
       const dim_t  m_iter,
       const dim_t  n_iter,
       const dim_t  k_iter,
       const dim_t  mr,
       const dim_t  nr,
             inc_t* j_st_p,
             inc_t* i_st_p,
             inc_t* j_en_p,
             inc_t* i_en_p
     )
{
	( void )mr;

	// Assumption: 0 <= diagoff. Make sure to prune leading rows beforehand!
	if ( diagoff < 0 ) bli_abort();

	// Single-threaded cases are simple and allow early returns.
	if ( nt == 1 )
	{
		const dim_t n_ut_for_me = m_iter * n_iter;

		*j_st_p = 0;
		*i_st_p = 0;
		*j_en_p = n_iter - 1;
		*i_en_p = m_iter - 1;

		return n_ut_for_me;
	}

	//
	// -- Step 1: Compute the computational volume of the region ---------------
	//

	// Normalize the diagonal offset by nr so that it represents the offset in
	// units of nr x nr chunks.
	const doff_t diagoff_iter = diagoff / nr;

	// For the purposes of many computations in this function, we aren't
	// interested in the extent to which diagoff exceeds n (if it does)
	// So we use a new variable that is guaranteed to be no greater than n.
	const doff_t diagoffmin_iter = bli_min( diagoff_iter, n_iter );

	const dim_t k_rect = k_iter;
	const dim_t n_rect = diagoffmin_iter;

	const dim_t gross_area   = k_rect * n_iter;
	const dim_t rect_area    = k_rect * n_rect;
	const dim_t nonrect_area = gross_area - rect_area;

	const dim_t offn_nonrect    = n_rect;
	const dim_t diagoff_nonrect = 0;

	const dim_t n_nonrect       = n_iter - n_rect;

	const dim_t offn_ut_nonrect = diagoffmin_iter;

	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "m_iter:             %7ld\n", (long) m_iter );
	PGUARD printf( "k_iter:             %7ld\n", (long) k_iter );
	PGUARD printf( "n_iter:             %7ld\n", (long) n_iter );
	PGUARD printf( "min(diagoff_it,n):  %7ld\n", (long) diagoffmin_iter );
	PGUARD printf( "offn_ut_nonrect:    %7ld\n", (long) offn_ut_nonrect );
	PGUARD printf( "offn_nonrect:       %7ld\n", (long) offn_nonrect );
	PGUARD printf( "diagoff_nonrect:    %7ld\n", (long) diagoff_nonrect );
	PGUARD printf( "n_nonrect:          %7ld\n", (long) n_nonrect );
	PGUARD printf( "---------------------------\n" );

	const dim_t num_unref_ut0 = n_nonrect * ( n_nonrect - 1 ) / 2;
	const dim_t num_unref_ut  = bli_max( 0, num_unref_ut0 );

	const dim_t tri_unref_area = num_unref_ut;
	const dim_t tri_ref_area   = nonrect_area - tri_unref_area;
	const dim_t total_ref_area = rect_area + tri_ref_area;
	const dim_t rect_vol       = rect_area * m_iter;
	const dim_t tri_ref_vol    = tri_ref_area * m_iter;
	const dim_t total_vol      = total_ref_area * m_iter;

	PGUARD printf( "gross_area:         %7ld\n", (long) gross_area );
	PGUARD printf( "nonrect_area:       %7ld\n", (long) nonrect_area );
	PGUARD printf( "tri_unref_area:     %7ld\n", (long) tri_unref_area );
	PGUARD printf( "rect_area:          %7ld\n", (long) rect_area );
	PGUARD printf( "tri_ref_area:       %7ld\n", (long) tri_ref_area );
	PGUARD printf( "total_ref_area:     %7ld\n", (long) total_ref_area );
	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "rect_vol (uops):    %7ld\n", (long) rect_vol );
	PGUARD printf( "tri_ref_vol (uops): %7ld\n", (long) tri_ref_vol );
	PGUARD printf( "total_vol (uops):   %7ld\n", (long) total_vol );
	PGUARD printf( "---------------------------\n" );

	//
	// -- Step 2: Compute key flop counts (per thread, per column, etc.) -------
	//

	//const dim_t rect_uops    = rect_vol;
	//const dim_t tri_ref_uops = tri_ref_vol;
	const dim_t total_uops   = total_vol;

	// Compute the number of microtile ops to allocate per thread as well as the
	// number of leftover microtile ops.
	const dim_t n_uops_per_thr = total_uops / nt;
	const dim_t n_uops_pt_left = total_uops % nt;

	PGUARD printf( "n_threads:          %7ld\n", (long) nt );
	PGUARD printf( "n_uops_per_thr:     %7ld\n", (long) n_uops_per_thr );
	PGUARD printf( "n_uops_pt_left:     %7ld\n", (long) n_uops_pt_left );
	PGUARD printf( "---------------------------\n" );

	const dim_t uops_per_col_rect = m_iter * k_iter;

	PGUARD printf( "uops_per_col_rect:  %7ld\n", (long) uops_per_col_rect );

	// Allocate one of the leftover uops to the current thread if its tid is
	// one of the lower thread ids.
	//const dim_t n_uops_for_me = n_uops_per_thr + ( tid < n_uops_pt_left ? 1 : 0 );

	//PGUARD printf( "n_uops_for_me:      %7ld (%ld+%ld)\n",
	//               n_uops_for_me, n_uops_per_thr, n_uops_for_me - n_uops_per_thr );

	//
	// -- Step 3: Compute the starting j/i utile offset for a given tid) -------
	//

	PGUARD printf( "---------------------------\n" );
	PGUARD printf( "total_utiles:       %7ld\n", (long) ( m_iter * n_iter ) );
	PGUARD printf( "---------------------------\n" );

	dim_t j_st_cur = 0; dim_t j_en_cur = 0;
	dim_t i_st_cur = 0; dim_t i_en_cur = 0;

	// Find the utile update that pushes uops_tba to 0 or less.
#ifdef PRINT_MODE
	for ( dim_t tid_i = 0; tid_i < nt; ++tid_i )
#else
	for ( dim_t tid_i = 0; tid_i < nt - 1; ++tid_i )
#endif
	{
		const dim_t uops_ta     = n_uops_per_thr + ( tid_i < n_uops_pt_left ? 1 : 0 );
		      dim_t uops_tba    = uops_ta;
		      dim_t j           = j_st_cur;
		      dim_t n_ut_for_me = 0;
		      bool  done_e      = FALSE;
		      bool  search_tri  = FALSE;

		PGUARD printf( "tid_i: %ld  n_uops_ta:    %3ld \n",
		               (long) tid_i, (long) uops_tba );
		PGUARD printf( "tid_i: %ld  j: %2ld  (  n_ut_cur: %ld) (uops_alloc: %ld)\n",
		               (long) tid_i, (long) j, (long) n_ut_for_me,
		               (long) ( uops_ta - uops_tba ) );

		// This code begins allocating uops when the starting point is somewhere
		// after the first microtile. Typically this will not be enough to
		// allocate all uops, except for situations where the number of threads
		// is high relative to the number of utile columns, in which case the
		// code signals an early finish (via done_e).
		if ( 0 < i_st_cur )
		{
			// Compute the number of uops needed to update each utile in the
			// current column.
			const dim_t k_iter_j = bli_tlb_trmm_rl_k_iter( diagoff_iter, k_iter, j );

			dim_t i;

			#if 0

			// Starting from i_st_cur within the current utile column, allocate
			// utiles until (a) we run out of utiles in the column (which is tyipcally
			// what happens), or (b) we finish allocating all uops for the current
			// thread (uops_tba drops to zero or less).
			for ( i = i_st_cur; i < m_iter; ++i )
			{
				n_ut_for_me += 1;

				const dim_t uops_tba_new = uops_tba - k_iter_j;

				uops_tba = uops_tba_new;

				PGUARD printf( "tid_i: %ld  i: %2ld  (0 n_ut_cur: %ld) (uops_alloc: %ld) (k_iter_j: %ld)\n",
				               (long) tid_i, (long) i, (long) n_ut_for_me,
				               (long) uops_ta - uops_tba, k_iter_j );

				if ( uops_tba_new <= 0 ) { j_en_cur = j; i_en_cur = i; done_e = TRUE;
				                           break; }
			}

			// If we traversed the entire column (regardless of whether we finished
			// allocating utiles for the current thread), increment j to the next
			// column, which is where we'll continue our search for the current tid
			// (or start our search for the next tid if we finished allocating utiles).
			// Additionally, if we finished traversing all utile columns, mark the
			// last utile of the last column as the end point, and set the "done early"
			// flag.
			if ( i == m_iter )
			{
				j += 1;
				if ( j == n_iter ) { j_en_cur = j - 1; i_en_cur = m_iter - 1; done_e = TRUE; }
			}

			#else

			// Compute the number of utiles left to allocate under the (probably false)
			// assumption that all utiles incur the same uop cost (k_iter_j) to update.
			// Also compute the number of utiles that remain in the current column.
			const dim_t n_ut_tba_j = uops_tba / k_iter_j + ( uops_tba % k_iter_j ? 1 : 0 );
			const dim_t n_ut_rem_j = m_iter - i_st_cur;

			// Compare the aforementioned values. If n_ut_tba_j is less than or equal to
			// the number of remaining utiles in the column, we can finish allocating
			// without moving to the next column. But if n_ut_tba_j exceeds n_ut_rem_j,
			// then we aren't done yet, so allocate what we can and move on.
			if ( n_ut_tba_j <= n_ut_rem_j )
			{
				n_ut_for_me += n_ut_tba_j;
				uops_tba    -= n_ut_tba_j * k_iter_j;
				i            = i_st_cur + n_ut_tba_j;

				j_en_cur = j; i_en_cur = i - 1; done_e = TRUE;
			}
			else // if ( n_ut_rem_j < n_ut_tba_j )
			{
				n_ut_for_me += n_ut_rem_j;
				uops_tba    -= n_ut_rem_j * k_iter_j;
				i            = i_st_cur + n_ut_rem_j;
			}

			PGUARD printf( "tid_i: %ld  i: %2ld  (* n_ut_cur: %ld) (uops_alloc: %ld)\n",
			               (long) tid_i, (long) i-1, (long) n_ut_for_me,
			               (long) ( uops_ta - uops_tba ) );

			// If we allocated all utiles in the column (regardless of whether we finished
			// allocating utiles for the current thread), increment j to the next column,
			// which is where we'll continue our search for the current tid's end point
			// (or start our search through the next tid's range if we finished allocating
			// the current tid's utiles). Additionally, if we allocated utiles from the
			// last column, mark the tid's end point and set the "done early" flag.
			if ( i == m_iter )
			{
				j += 1; i = 0;
				if ( j == n_iter ) { j_en_cur = j - 1; i_en_cur = m_iter - 1; done_e = TRUE; }

				PGUARD printf( "tid_i: %ld  j: %2ld  (! n_ut_cur: %ld) (uops_alloc: %ld)\n",
				               (long) tid_i, (long) j, (long) n_ut_for_me,
				               (long) ( uops_ta - uops_tba ) );
			}

			#endif
		}

		// This code advances over as many columns of utiles as possible, within
		// the rectangular region (i.e., pre-diagonal), and then walks down to
		// the correct utile within the subsequent column. However, note that
		// this code gets skipped entirely if the previous code block was able
		// to allocate all of the current tid's uops.
		if ( !done_e )
		{
			// If j is positioned somewhere within the rectangular region, we can
			// skip over as many utile columns as possible with some integer math.
			// And depending on how many uops we were able to allocate relative to
			// the number of columns that exist, we may need to walk through the
			// triangular region as well. But if j is already in the triangular
			// region, we set a flag so that we execute the code that will walk
			// through those columns.
			if ( j < diagoff_iter )
			{
				const dim_t j_inc0  = uops_tba / uops_per_col_rect;
				const dim_t j_left0 = uops_tba % uops_per_col_rect;

				// We need to set a hard limit on how much j_inc can be. Namely,
				// it should not exceed the number of utile columns that are left
				// in the rectangular region of the matrix, nor should it exceed
				// the total number of utile columns that are left.
				const dim_t j_inc1 = bli_min( j_inc0, diagoff_iter - j );
				const dim_t j_inc  = bli_min( j_inc1, n_iter - j );
				const dim_t delta  = j_inc0 - j_inc;
				const dim_t j_left = j_left0 + delta * uops_per_col_rect;

				// Increment j by the number of full utile columns we allocate, and
				// set the remaining utile ops to be allocated to the remainder.
				j       += j_inc;
				uops_tba = j_left;

				n_ut_for_me += j_inc * m_iter;

				PGUARD printf( "tid_i: %ld  advanced to col: %2ld  (uops traversed: %ld)\n",
				               (long) tid_i, (long) j, (long) ( uops_per_col_rect * j_inc ) );
				PGUARD printf( "tid_i: %ld  j: %2ld  (1 n_ut_cur: %ld) (uops_alloc: %ld)\n",
				               (long) tid_i, (long) j, (long) n_ut_for_me,
				               (long) ( uops_ta - uops_tba ) );
				PGUARD printf( "tid_i: %ld  uops left to alloc: %2ld \n",
				               (long) tid_i, (long) j_left );

				if ( uops_tba == 0 )
				{
					// If advancing j_inc columns allocated all of our uops, then
					// designate the last iteration of the previous column as the
					// end point.
					j_en_cur = j - 1;
					i_en_cur = m_iter - 1;
					search_tri = FALSE;

					PGUARD printf( "tid_i: %ld  j: %2ld  (2 n_ut_cur: %ld) (uops_alloc: %ld)\n",
					               (long) tid_i, (long) j, (long) n_ut_for_me,
					               (long) ( uops_ta - uops_tba ) );
				}
				else if ( j >  n_iter ) bli_abort(); // Safety check; should never execute.
				else if ( j == n_iter )
				{
					// If we still have at least some uops to allocate, and advancing
					// j_inc columns landed us at the beginning of the first non-
					// existent column (column n_iter), then we're done. (The fact
					// that we didn't get to allocate all of our uops just means that
					// the lower tids slightly overshot their allocations, leaving
					// fewer uops for the last thread.)
					search_tri = FALSE;
					PGUARD printf( "tid_i: %ld  j: %2ld  (3 n_ut_cur: %ld) (uops_alloc: %ld)\n",
					               (long) tid_i, (long) j, (long) n_ut_for_me,
					               (long) ( uops_ta - uops_tba ) );
				}
				else if ( j < diagoff_iter )
				{
					// If we still have at least some uops to allocate, and advancing
					// j_inc columns landed us at the beginning of a column that is
					// still in the rectangular region, then we don't need to enter
					// the triangular region (if it even exists). The code below will
					// walk down the current column and find the utile that puts us
					// over the top.
					search_tri = FALSE;
					PGUARD printf( "tid_i: %ld  j: %2ld  (4 n_ut_cur: %ld) (uops_alloc: %ld)\n",
					               (long) tid_i, (long) j, (long)  n_ut_for_me,
					               (long) ( uops_ta - uops_tba ) );
				}
				else // if ( 0 < uops_tba && j == diagoff_iter && j < n_iter )
				{
					// If we have at least some uops to allocate, and we still have
					// at least some columns to process, then we set a flag to
					// indicate that we still need to step through the triangular
					// region.
					search_tri = TRUE;
					PGUARD printf( "tid_i: %ld  j: %2ld  (5 n_ut_cur: %ld) (uops_alloc: %ld)\n",
					               (long) tid_i, (long) j, (long) n_ut_for_me,
					               (long) ( uops_ta - uops_tba ) );
				}
			}
			else /* if ( diagoff_iter <= j ) */
			{
				PGUARD printf( "tid_i: %ld  j: %2ld >= diagoff_iter: %ld\n",
				               (long) tid_i, (long) j, (long) diagoff_iter );
				search_tri = TRUE;
			}

			PGUARD printf( "tid_i: %ld  j: %2ld  search_tri: %ld\n", (long) tid_i,
			               (long) j, (long) search_tri );

			if ( search_tri )
			{
				// If we still have some uops to allocate in the triangular region,
				// we first allocate as many full utile columns as possible without
				// exceeding the number of uops left to be allocated.
				for ( ; j < n_iter; ++j )
				{
					const dim_t k_iter_j = bli_tlb_trmm_rl_k_iter( diagoff_iter, k_iter, j );
					const dim_t n_uops_j = k_iter_j * m_iter;

					PGUARD printf( "tid_i: %ld  j: %2ld  (6 n_ut_cur: %ld) (uops_alloc: %ld) (n_uops_j: %ld)\n",
					               (long) tid_i, (long) j, (long) n_ut_for_me,
					               (long) ( uops_ta - uops_tba ), (long) n_uops_j );

					if ( uops_tba == 0 )
					{
						PGUARD printf( "tid_i: %ld  j: %2ld  (7 n_ut_cur: %ld) (uops_alloc: %ld)\n",
						               (long) tid_i, (long) j, (long) n_ut_for_me, (long) ( uops_ta - uops_tba ) );
						// If advancing over the previous column allocated all of
						// our uops, then designate the last iteration of the
						// previous column as the end point.
						j_en_cur = j - 1;
						i_en_cur = m_iter - 1;
						break;
					}
					if ( n_uops_j <= uops_tba )
					{
						// If advancing over the current column doesn't exceed the
						// number of uops left to allocate, then allocate them. (If
						// n_uops_j == uops_tba, then we'll be done shortly after
						// incrementing j.)
						n_ut_for_me += m_iter;
						uops_tba -= n_uops_j;

						PGUARD printf( "tid_i: %ld  j: %2ld  (8 n_ut_cur: %ld) (uops_alloc: %ld)\n",
						               (long) tid_i, (long) j, (long) n_ut_for_me,
						               (long) ( uops_ta - uops_tba ) );
					}
					else // if ( uops_tba < n_uops_j )
					{
						PGUARD printf( "tid_i: %ld  j: %2ld  (9 n_ut_cur: %ld) (uops_alloc: %ld)\n",
						               (long) tid_i, (long) j, (long) n_ut_for_me,
						               (long) ( uops_ta - uops_tba ) );
						// If we can finish allocating all the remaining uops
						// with the utiles in the current column, then we break
						// out of the loop without updating j, n_ut_for_me, or
						// uops_tba. The remaining uops will be allocated in
						// the loop over m_iter below.
						break;
					}
				}
			}

			// If there are any uops left to allocate, and we haven't already
			// exhausted all allocatable utiles, it means that we have to walk down
			// the current column and find the utile that puts us over the top.
			if ( 0 < uops_tba && j < n_iter )
			{
				const dim_t k_iter_j = bli_tlb_trmm_rl_k_iter( diagoff_iter, k_iter, j );

				PGUARD printf( "tid_i: %ld  j: %2ld  (A n_ut_cur: %ld) (uops_alloc: %ld) (k_iter_j: %ld)\n",
				               (long) tid_i, (long) j, (long) n_ut_for_me,
				               (long) ( uops_ta - uops_tba ), (long) k_iter_j );

				#if 0

				dim_t i;
				for ( i = 0; i < m_iter; ++i )
				{
					n_ut_for_me += 1;
					const dim_t uops_tba_new = uops_tba - k_iter_j;
					uops_tba = uops_tba_new;
					PGUARD printf( "tid_i: %ld  i: %2ld  (B n_ut_cur: %ld) (uops_alloc: %ld)\n",
					               (long) tid_i, (long) i, (long) n_ut_for_me, uops_ta - uops_tba );
					if ( uops_tba_new <= 0 ) { j_en_cur = j; i_en_cur = i; break; }
				}

				if ( i == m_iter )
				{
					j += 1;
					if ( j == n_iter ) { j_en_cur = j - 1; i_en_cur = m_iter - 1; }
				}

				#else

				const dim_t n_ut_j = uops_tba / k_iter_j + ( uops_tba % k_iter_j ? 1 : 0 );
				const dim_t i      = n_ut_j - 1;

				uops_tba    -= n_ut_j * k_iter_j;
				n_ut_for_me += n_ut_j;

				j_en_cur = j; i_en_cur = i;

				PGUARD printf( "tid_i: %ld  i: %2ld  (b n_ut_cur: %ld) (uops_alloc: %ld)\n",
				               (long) tid_i, (long) i, (long) n_ut_for_me,
				               (long) ( uops_ta - uops_tba ) );

				#endif
			}
			else // if ( uops_tba <= 0 || j == n_iter )
			{
				j_en_cur = j - 1;
				i_en_cur = m_iter - 1;
			}
		}

		PGUARD printf( "tid_i: %ld  done!  (C n_ut_cur: %ld) (overshoot: %ld out of %ld)\n",
		               (long) tid_i, (long) n_ut_for_me, -(long) uops_tba, (long) uops_ta );

		if ( tid_i == tid )
		{
			*j_st_p = j_st_cur;
			*i_st_p = i_st_cur;
			*j_en_p = j_en_cur;
			*i_en_p = i_en_cur;
			return n_ut_for_me;
		}

		// Use the current tid's ending i,j values to determine the starting i,j
		// values for the next tid.
		j_st_cur = j_en_cur;
		i_st_cur = i_en_cur + 1;
		if ( i_st_cur == m_iter ) { j_st_cur += 1; i_st_cur = 0; }

		PGUARD printf( "tid_i: %ld         (D n_ut_cur: %ld)\n",
		               (long) tid_i, (long) n_ut_for_me );
		PGUARD printf( "tid_i: %ld  tid %ld will start at j,i: %ld %ld\n",
		               (long) tid_i, (long) tid_i + 1,
		               (long) j_st_cur, (long) i_st_cur );
		PGUARD printf( "---------------------------\n" );
	}

#ifndef PRINT_MODE

	//
	// -- Step 4: Handle the last thread's allocation --------------------------
	//

	// An optimization: The above loop runs to nt - 1 rather than nt since it's
	// easy to count the number of utiles allocated to the last thread.
	const dim_t n_ut_for_me = m_iter - i_st_cur +
	                          (n_iter - j_st_cur - 1) * m_iter;
	*j_st_p = j_st_cur;
	*i_st_p = i_st_cur;
	*j_en_p = n_iter - 1;
	*i_en_p = m_iter - 1;

	PGUARD printf( "tid_i: %ld         (E n_ut_for_me: %ld) (j,i_st: %ld %ld)\n",
	               (long) tid, (long) n_ut_for_me,
	               (long) j_st_cur, (long) i_st_cur );

	return n_ut_for_me;
#else
	// This line should never execute, but we need it to satisfy the compiler.
	return -1;
#endif
}
