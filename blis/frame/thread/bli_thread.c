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

// A global communicator that is hard-coded for single-threaded execution.
thrcomm_t BLIS_SINGLE_COMM = {};

// -----------------------------------------------------------------------------

typedef void (*thread_launch_t)
     (
             dim_t         nt,
             thread_func_t func,
       const void*         params
     );

static thread_launch_t thread_launch_fpa[ BLIS_NUM_THREAD_IMPLS ] =
{
	[BLIS_SINGLE] = bli_thread_launch_single,
	[BLIS_OPENMP] =
#if   defined(BLIS_ENABLE_OPENMP)
	                bli_thread_launch_openmp,
#else
	                NULL,
#endif
	[BLIS_POSIX]  =
#if   defined(BLIS_ENABLE_PTHREADS)
	                bli_thread_launch_pthreads,
#else
	                NULL,
#endif
	[BLIS_HPX] =
#if   defined(BLIS_ENABLE_HPX)
	                bli_thread_launch_hpx,
#else
	                NULL,
#endif
};

// -----------------------------------------------------------------------------

int bli_thread_init( void )
{
	// NOTE: This function is called once by ONLY ONE application thread per
	// library init/finalize cycle (see bli_init.c). Thus, a mutex is not
	// needed to protect the data initialization.

	bli_thrcomm_init( BLIS_SINGLE, 1, &BLIS_SINGLE_COMM );

	return 0;
}

int bli_thread_finalize( void )
{
	bli_thrcomm_cleanup( &BLIS_SINGLE_COMM );

	return 0;
}

// -----------------------------------------------------------------------------

void bli_thread_launch
     (
             timpl_t       ti,
             dim_t         nt,
             thread_func_t func,
       const void*         params
     )
{
	thread_launch_fpa[ti]( nt, func, params );
}

// -----------------------------------------------------------------------------

void bli_prime_factorization( dim_t n, bli_prime_factors_t* factors )
{
	factors->n = n;
	factors->sqrt_n = ( dim_t )sqrt( ( double )n );
	factors->f = 2;
}

dim_t bli_next_prime_factor( bli_prime_factors_t* factors )
{
	// Return the prime factorization of the original number n one-by-one.
	// Return 1 after all factors have been exhausted.

	// Looping over possible factors in increasing order assures we will
	// only return prime factors (a la the Sieve of Eratosthenes).
	while ( factors->f <= factors->sqrt_n )
	{
		// Special cases for factors 2-7 handle all numbers not divisible by 11
		// or another larger prime. The slower loop version is used after that.
		// If you use a number of threads with large prime factors you get
		// what you deserve.
		if ( factors->f == 2 )
		{
			if ( factors->n % 2 == 0 )
			{
				factors->n /= 2;
				return 2;
			}
			factors->f = 3;
		}
		else if ( factors->f == 3 )
		{
			if ( factors->n % 3 == 0 )
			{
				factors->n /= 3;
				return 3;
			}
			factors->f = 5;
		}
		else if ( factors->f == 5 )
		{
			if ( factors->n % 5 == 0 )
			{
				factors->n /= 5;
				return 5;
			}
			factors->f = 7;
		}
		else if ( factors->f == 7 )
		{
			if ( factors->n % 7 == 0 )
			{
				factors->n /= 7;
				return 7;
			}
			factors->f = 11;
		}
		else
		{
			if ( factors->n % factors->f == 0 )
			{
				factors->n /= factors->f;
				return factors->f;
			}
			factors->f++;
		}
	}

	// To get here we must be out of prime factors, leaving only n (if it is
	// prime) or an endless string of 1s.
	dim_t tmp = factors->n;
	factors->n = 1;
	return tmp;
}

bool bli_is_prime( dim_t n )
{
	bli_prime_factors_t factors;

	bli_prime_factorization( n, &factors );

	dim_t f = bli_next_prime_factor( &factors );

	if ( f == n ) return TRUE;
	else          return FALSE;
}

void bli_thread_partition_2x2
     (
       dim_t           n_thread,
       dim_t           work1,
       dim_t           work2,
       dim_t* restrict nt1,
       dim_t* restrict nt2
     )
{
	// Partition a number of threads into two factors nt1 and nt2 such that
	// nt1/nt2 ~= work1/work2. There is a fast heuristic algorithm and a
	// slower optimal algorithm (which minimizes |nt1*work2 - nt2*work1|).

	// Return early small prime numbers of threads.
	if ( n_thread < 4 )
	{
		*nt1 = ( work1 >= work2 ? n_thread : 1 );
		*nt2 = ( work1 <  work2 ? n_thread : 1 );

		return;
	}

#if 1
	bli_thread_partition_2x2_fast( n_thread, work1, work2, nt1, nt2 );
#else
	bli_thread_partition_2x2_slow( n_thread, work1, work2, nt1, nt2 );
#endif
}

//#define PRINT_FACTORS

void bli_thread_partition_2x2_fast
     (
       dim_t           n_thread,
       dim_t           work1,
       dim_t           work2,
       dim_t* restrict nt1,
       dim_t* restrict nt2
     )
{
	// Compute with these local variables until the end of the function, at
	// which time we will save the values back to nt1 and nt2.
	dim_t tn1 = 1;
	dim_t tn2 = 1;

	// Both algorithms need the prime factorization of n_thread.
	bli_prime_factors_t factors;
	bli_prime_factorization( n_thread, &factors );

	// Fast algorithm: assign prime factors in increasing order to whichever
	// partition has more work to do. The work is divided by the number of
	// threads assigned at each iteration. This algorithm is sub-optimal in
	// some cases. We attempt to mitigate the cases that involve at least one
	// factor of 2. For example, in the partitioning of 12 with equal work
	// this algorithm tentatively finds 6x2. This factorization involves a
	// factor of 2 that can be reallocated, allowing us to convert it to the
	// optimal solution of 4x3. But some cases cannot be corrected this way
	// because they do not contain a factor of 2. For example, this algorithm
	// factors 105 (with equal work) into 21x5 whereas 7x15 would be optimal.

	#ifdef PRINT_FACTORS
	printf( "w1 w2 = %d %d (initial)\n", (int)work1, (int)work2 );
	#endif

	dim_t f;
	while ( ( f = bli_next_prime_factor( &factors ) ) > 1 )
	{
		#ifdef PRINT_FACTORS
		printf( "w1 w2 = %4d %4d nt1 nt2 = %d %d ... f = %d\n",
		        (int)work1, (int)work2, (int)tn1, (int)tn2, (int)f );
		#endif

		if ( work1 > work2 ) { work1 /= f; tn1 *= f; }
		else                 { work2 /= f; tn2 *= f; }
	}

	#ifdef PRINT_FACTORS
	printf( "w1 w2 = %4d %4d nt1 nt2 = %d %d\n",
	        (int)work1, (int)work2, (int)tn1, (int)tn2 );
	#endif

	// Sometimes the last factor applied is prime. For example, on a square
	// matrix, we tentatively arrive (from the logic above) at:
	// - a 2x6 factorization when given 12 ways of parallelism
	// - a 2x10 factorization when given 20 ways of parallelism
	// - a 2x14 factorization when given 28 ways of parallelism
	// These factorizations are suboptimal under the assumption that we want
	// the parallelism to be as balanced as possible. Below, we make a final
	// attempt at rebalancing nt1 and nt2 by checking to see if the gap between
	// work1 and work2 is narrower if we reallocate a factor of 2.
	if ( work1 > work2 )
	{
		// Example: nt = 12
		//          w1 w2 (initial)   = 3600 3600; nt1 nt2 =  1 1
		//          w1 w2 (tentative) = 1800  600; nt1 nt2 =  2 6
		//          w1 w2 (ideal)     =  900 1200; nt1 nt2 =  4 3
		if ( tn2 % 2 == 0 )
		{
			dim_t diff     =          work1   - work2;
			dim_t diff_mod = bli_abs( work1/2 - work2*2 );

			if ( diff_mod < diff ) { tn1 *= 2; tn2 /= 2; }
		}
	}
	else if ( work1 < work2 )
	{
		// Example: nt = 40
		//          w1 w2 (initial)   = 3600 3600; nt1 nt2 =  1 1
		//          w1 w2 (tentative) =  360  900; nt1 nt2 = 10 4
		//          w1 w2 (ideal)     =  720  450; nt1 nt2 =  5 8
		if ( tn1 % 2 == 0 )
		{
			dim_t diff     =          work2   - work1;
			dim_t diff_mod = bli_abs( work2/2 - work1*2 );

			if ( diff_mod < diff ) { tn1 /= 2; tn2 *= 2; }
		}
	}

	#ifdef PRINT_FACTORS
	printf( "w1 w2 = %4d %4d nt1 nt2 = %d %d (final)\n",
	        (int)work1, (int)work2, (int)tn1, (int)tn2 );
	#endif

	// Save the final result.
	*nt1 = tn1;
	*nt2 = tn2;
}

#include "limits.h"

void bli_thread_partition_2x2_slow
     (
       dim_t           n_thread,
       dim_t           work1,
       dim_t           work2,
       dim_t* restrict nt1,
       dim_t* restrict nt2
     )
{
	// Slow algorithm: exhaustively constructs all factor pairs of n_thread and
	// chooses the best one.

	// Compute with these local variables until the end of the function, at
	// which time we will save the values back to nt1 and nt2.
	dim_t tn1 = 1;
	dim_t tn2 = 1;

	// Both algorithms need the prime factorization of n_thread.
	bli_prime_factors_t factors;
	bli_prime_factorization( n_thread, &factors );

	// Eight prime factors handles n_thread up to 223092870.
	dim_t fact[8];
	dim_t mult[8];

	// There is always at least one prime factor, so use if for initialization.
	dim_t nfact = 1;
	fact[0] = bli_next_prime_factor( &factors );
	mult[0] = 1;

	// Collect the remaining prime factors, accounting for multiplicity of
	// repeated factors.
	dim_t f;
	while ( ( f = bli_next_prime_factor( &factors ) ) > 1 )
	{
		if ( f == fact[nfact-1] )
		{
			mult[nfact-1]++;
		}
		else
		{
			nfact++;
			fact[nfact-1] = f;
			mult[nfact-1] = 1;
		}
	}

	// Now loop over all factor pairs. A single factor pair is denoted by how
	// many of each prime factor are included in the first factor (ntaken).
	dim_t ntake[8] = {0};
	dim_t min_diff = INT_MAX;

	// Loop over how many prime factors to assign to the first factor in the
	// pair, for each prime factor. The total number of iterations is
	// \Prod_{i=0}^{nfact-1} mult[i].
	bool done = FALSE;
	while ( !done )
	{
		dim_t x = 1;
		dim_t y = 1;

		// Form the factors by integer exponentiation and accumulation.
		for ( dim_t i = 0 ; i < nfact ; i++ )
		{
			x *= bli_ipow( fact[i], ntake[i] );
			y *= bli_ipow( fact[i], mult[i]-ntake[i] );
		}

		// Check if this factor pair is optimal by checking
		// |nt1*work2 - nt2*work1|.
		dim_t diff = llabs( x*work2 - y*work1 );
		if ( diff < min_diff )
		{
			min_diff = diff;
			tn1 = x;
			tn2 = y;
		}

		// Go to the next factor pair by doing an "odometer loop".
		for ( dim_t i = 0 ; i < nfact ; i++ )
		{
			if ( ++ntake[i] > mult[i] )
			{
				ntake[i] = 0;
				if ( i == nfact-1 ) done = TRUE;
				else continue;
			}
			break;
		}
	}

	// Save the final result.
	*nt1 = tn1;
	*nt2 = tn2;
}

#if 0
void bli_thread_partition_2x2_orig
     (
       dim_t           n_thread,
       dim_t           work1,
       dim_t           work2,
       dim_t* restrict nt1,
       dim_t* restrict nt2
     )
{
	// Copy nt1 and nt2 to local variables and then compute with those local
	// variables until the end of the function, at which time we will save the
	// values back to nt1 and nt2.
	dim_t tn1; // = *nt1;
	dim_t tn2; // = *nt2;

	// Partition a number of threads into two factors nt1 and nt2 such that
	// nt1/nt2 ~= work1/work2. There is a fast heuristic algorithm and a
	// slower optimal algorithm (which minimizes |nt1*work2 - nt2*work1|).

	// Return early small prime numbers of threads.
	if ( n_thread < 4 )
	{
		tn1 = ( work1 >= work2 ? n_thread : 1 );
		tn2 = ( work1 <  work2 ? n_thread : 1 );

		return;
	}

	tn1 = 1;
	tn2 = 1;

	// Both algorithms need the prime factorization of n_thread.
	bli_prime_factors_t factors;
	bli_prime_factorization( n_thread, &factors );

#if 1

	// Fast algorithm: assign prime factors in increasing order to whichever
	// partition has more work to do. The work is divided by the number of
	// threads assigned at each iteration. This algorithm is sub-optimal in
	// some cases. We attempt to mitigate the cases that involve at least one
	// factor of 2. For example, in the partitioning of 12 with equal work
	// this algorithm tentatively finds 6x2. This factorization involves a
	// factor of 2 that can be reallocated, allowing us to convert it to the
	// optimal solution of 4x3. But some cases cannot be corrected this way
	// because they do not contain a factor of 2. For example, this algorithm
	// factors 105 (with equal work) into 21x5 whereas 7x15 would be optimal.

	//printf( "w1 w2 = %d %d (initial)\n", (int)work1, (int)work2 );

	dim_t f;
	while ( ( f = bli_next_prime_factor( &factors ) ) > 1 )
	{
		//printf( "w1 w2 = %4d %4d nt1 nt2 = %d %d ... f = %d\n", (int)work1, (int)work2, (int)tn1, (int)tn2, (int)f );

		if ( work1 > work2 )
		{
			work1 /= f;
			tn1 *= f;
		}
		else
		{
			work2 /= f;
			tn2 *= f;
		}
	}

	//printf( "w1 w2 = %4d %4d nt1 nt2 = %d %d\n", (int)work1, (int)work2, (int)tn1, (int)tn2 );

	// Sometimes the last factor applied is prime. For example, on a square
	// matrix, we tentatively arrive (from the logic above) at:
	// - a 2x6 factorization when given 12 ways of parallelism
	// - a 2x10 factorization when given 20 ways of parallelism
	// - a 2x14 factorization when given 28 ways of parallelism
	// These factorizations are suboptimal under the assumption that we want
	// the parallelism to be as balanced as possible. Below, we make a final
	// attempt at rebalancing nt1 and nt2 by checking to see if the gap between
	// work1 and work2 is narrower if we reallocate a factor of 2.
	if ( work1 > work2 )
	{
		// Example: nt = 12
		//          w1 w2 (initial)   = 3600 3600; nt1 nt2 =  1 1
		//          w1 w2 (tentative) = 1800  600; nt1 nt2 =  2 6
		//          w1 w2 (ideal)     =  900 1200; nt1 nt2 =  4 3
		if ( tn2 % 2 == 0 )
		{
			dim_t diff     =          work1   - work2;
			dim_t diff_mod = bli_abs( work1/2 - work2*2 );

			if ( diff_mod < diff ) { tn1 *= 2; tn2 /= 2; }
		}
	}
	else if ( work1 < work2 )
	{
		// Example: nt = 40
		//          w1 w2 (initial)   = 3600 3600; nt1 nt2 =  1 1
		//          w1 w2 (tentative) =  360  900; nt1 nt2 = 10 4
		//          w1 w2 (ideal)     =  720  450; nt1 nt2 =  5 8
		if ( tn1 % 2 == 0 )
		{
			dim_t diff     =          work2   - work1;
			dim_t diff_mod = bli_abs( work2/2 - work1*2 );

			if ( diff_mod < diff ) { tn1 /= 2; tn2 *= 2; }
		}
	}

	//printf( "w1 w2 = %4d %4d nt1 nt2 = %d %d (final)\n", (int)work1, (int)work2, (int)tn1, (int)tn2 );

#else

	// Slow algorithm: exhaustively constructs all factor pairs of n_thread and
	// chooses the best one.

	// Eight prime factors handles n_thread up to 223092870.
	dim_t fact[8];
	dim_t mult[8];

	// There is always at least one prime factor, so use if for initialization.
	dim_t nfact = 1;
	fact[0] = bli_next_prime_factor( &factors );
	mult[0] = 1;

	// Collect the remaining prime factors, accounting for multiplicity of
	// repeated factors.
	dim_t f;
	while ( ( f = bli_next_prime_factor( &factors ) ) > 1 )
	{
		if ( f == fact[nfact-1] )
		{
			mult[nfact-1]++;
		}
		else
		{
			nfact++;
			fact[nfact-1] = f;
			mult[nfact-1] = 1;
		}
	}

	// Now loop over all factor pairs. A single factor pair is denoted by how
	// many of each prime factor are included in the first factor (ntaken).
	dim_t ntake[8] = {0};
	dim_t min_diff = INT_MAX;

	// Loop over how many prime factors to assign to the first factor in the
	// pair, for each prime factor. The total number of iterations is
	// \Prod_{i=0}^{nfact-1} mult[i].
	bool   done = FALSE;
	while ( !done )
	{
		dim_t x = 1;
		dim_t y = 1;

		// Form the factors by integer exponentiation and accumulation.
		for  (dim_t i = 0 ; i < nfact ; i++ )
		{
			x *= bli_ipow( fact[i], ntake[i] );
			y *= bli_ipow( fact[i], mult[i]-ntake[i] );
		}

		// Check if this factor pair is optimal by checking
		// |nt1*work2 - nt2*work1|.
		dim_t diff = llabs( x*work2 - y*work1 );
		if ( diff < min_diff )
		{
			min_diff = diff;
			tn1 = x;
			tn2 = y;
		}

		// Go to the next factor pair by doing an "odometer loop".
		for ( dim_t i = 0 ; i < nfact ; i++ )
		{
			if ( ++ntake[i] > mult[i] )
			{
				ntake[i] = 0;
				if ( i == nfact-1 ) done = TRUE;
				else continue;
			}
			break;
		}
	}

#endif


	// Save the final result.
	*nt1 = tn1;
	*nt2 = tn2;
}
#endif

// -----------------------------------------------------------------------------

dim_t bli_gcd( dim_t x, dim_t y )
{
	while ( y != 0 )
	{
		dim_t t = y;
		y = x % y;
		x = t;
	}
	return x;
}

dim_t bli_lcm( dim_t x, dim_t y)
{
	return x * y / bli_gcd( x, y );
}

dim_t bli_ipow( dim_t base, dim_t power )
{
	dim_t p = 1;

	for ( dim_t mask = 0x1 ; mask <= power ; mask <<= 1 )
	{
		if ( power & mask ) p *= base;
		base *= base;
	}

	return p;
}

// -----------------------------------------------------------------------------

dim_t bli_thread_get_jc_nt( void )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	return bli_rntm_jc_ways( bli_global_rntm() );
}

dim_t bli_thread_get_pc_nt( void )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	return bli_rntm_pc_ways( bli_global_rntm() );
}

dim_t bli_thread_get_ic_nt( void )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	return bli_rntm_ic_ways( bli_global_rntm() );
}

dim_t bli_thread_get_jr_nt( void )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	return bli_rntm_jr_ways( bli_global_rntm() );
}

dim_t bli_thread_get_ir_nt( void )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	return bli_rntm_ir_ways( bli_global_rntm() );
}

dim_t bli_thread_get_num_threads( void )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	return bli_rntm_num_threads( bli_global_rntm() );
}

timpl_t bli_thread_get_thread_impl( void )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	return bli_rntm_thread_impl( bli_global_rntm() );
}

static const char* bli_timpl_string[BLIS_NUM_THREAD_IMPLS] =
{
	[BLIS_SINGLE] = "single",
	[BLIS_OPENMP] = "openmp",
	[BLIS_POSIX]  = "pthreads",
	[BLIS_HPX]    = "hpx",
};

const char* bli_thread_get_thread_impl_str( timpl_t ti )
{
	return bli_timpl_string[ti];
}

// ----------------------------------------------------------------------------

void bli_thread_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

#ifdef BLIS_ENABLE_MULTITHREADING

	// If TLS is disabled, we need to use a mutex to protect the global rntm_t
	// since it will be shared with all application threads.
	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_lock( bli_global_rntm_mutex() );
	#endif

	bli_rntm_set_ways_only( jc, 1, ic, jr, ir, bli_global_rntm() );

	// Ensure that the rntm_t is in a consistent state.
	bli_rntm_sanitize( bli_global_rntm() );

	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_unlock( bli_global_rntm_mutex() );
	#endif

#else

	// When multithreading is disabled at compile time, ignore the user's
	// request.

#endif
}

void bli_thread_set_num_threads( dim_t n_threads )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

#ifdef BLIS_ENABLE_MULTITHREADING

	// If TLS is disabled, we need to use a mutex to protect the global rntm_t
	// since it will be shared with all application threads.
	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_lock( bli_global_rntm_mutex() );
	#endif

	bli_rntm_set_num_threads_only( n_threads, bli_global_rntm() );

	// Ensure that the rntm_t is in a consistent state.
	bli_rntm_sanitize( bli_global_rntm() );

	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_unlock( bli_global_rntm_mutex() );
	#endif

#else

	// When multithreading is disabled at compile time, ignore the user's
	// request.

#endif
}

void bli_thread_set_thread_impl( timpl_t ti )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	// If TLS is disabled, we need to use a mutex to protect the global rntm_t
	// since it will be shared with all application threads.
	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_lock( bli_global_rntm_mutex() );
	#endif

	bli_rntm_set_thread_impl_only( ti, bli_global_rntm() );

	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_unlock( bli_global_rntm_mutex() );
	#endif
}

void bli_thread_reset( void )
{
	// We must ensure that global_rntm_at_init has been initialized.
	bli_init_once();

	// If TLS is disabled, we need to use a mutex to protect the global rntm_t
	// since it will be shared with all application threads.
	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_lock( bli_global_rntm_mutex() );
	#endif

	// Overwrite the global rntm_t with the contents of the snapshot we took
	// at initialization.

	rntm_t* src = bli_global_rntm_at_init();
	rntm_t* dst = bli_global_rntm();

	timpl_t ti = bli_rntm_thread_impl( src );
	bool    af = bli_rntm_auto_factor( src );
	dim_t   nt = bli_rntm_num_threads( src );

	bli_rntm_set_thread_impl_only( ti, dst );
	bli_rntm_set_auto_factor_only( af, dst );
	bli_rntm_set_num_threads_only( nt, dst );

	dim_t   jc = bli_rntm_jc_ways( src );
	dim_t   pc = bli_rntm_pc_ways( src );
	dim_t   ic = bli_rntm_ic_ways( src );
	dim_t   jr = bli_rntm_jr_ways( src );
	dim_t   ir = bli_rntm_ir_ways( src );

	bli_rntm_set_ways_only( jc, pc, ic, jr, ir, dst );

	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_unlock( bli_global_rntm_mutex() );
	#endif
}

