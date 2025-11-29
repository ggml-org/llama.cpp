/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
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

#ifndef BLIS_RNTM_H
#define BLIS_RNTM_H


// Runtime object type (defined in bli_type_defs.h)

/*
typedef struct rntm_s
{
	timpl_t   thread_impl;

	dim_t     num_threads;
	dim_t     thrloop[ BLIS_NUM_LOOPS ];

	bool      auto_factor;
	bool      pack_a;
	bool      pack_b;
	bool      l3_sup;
} rntm_t;
*/

//
// -- rntm_t query (public API) ------------------------------------------------
//

BLIS_INLINE timpl_t bli_rntm_thread_impl( const rntm_t* rntm )
{
	return rntm->thread_impl;
}

BLIS_INLINE bool bli_rntm_auto_factor( const rntm_t* rntm )
{
	return rntm->auto_factor;
}

BLIS_INLINE dim_t bli_rntm_num_threads( const rntm_t* rntm )
{
	return rntm->num_threads;
}

BLIS_INLINE dim_t bli_rntm_ways_for( bszid_t bszid, const rntm_t* rntm )
{
	return ( bszid == BLIS_NO_PART ? 1 : rntm->thrloop[ bszid ] );
}

BLIS_INLINE dim_t bli_rntm_total_ways_for( dim_t thread_by, const rntm_t* rntm )
{
    dim_t n_way = 1;
    for ( dim_t i = 0; i < BLIS_NUM_LOOPS; i++ )
	{
        if ( thread_by & (1 << i) )
            n_way *= bli_rntm_ways_for( (bszid_t)i, rntm );
    }
    return n_way;
}

BLIS_INLINE dim_t bli_rntm_jc_ways( const rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_NC, rntm );
}
BLIS_INLINE dim_t bli_rntm_pc_ways( const rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_KC, rntm );
}
BLIS_INLINE dim_t bli_rntm_ic_ways( const rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_MC, rntm );
}
BLIS_INLINE dim_t bli_rntm_jr_ways( const rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_NR, rntm );
}
BLIS_INLINE dim_t bli_rntm_ir_ways( const rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_MR, rntm );
}
BLIS_INLINE dim_t bli_rntm_pr_ways( const rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_KR, rntm );
}

BLIS_INLINE bool bli_rntm_pack_a( const rntm_t* rntm )
{
	return ( bool )( rntm->pack_a );
}
BLIS_INLINE bool bli_rntm_pack_b( const rntm_t* rntm )
{
	return ( bool )( rntm->pack_b );
}

BLIS_INLINE bool bli_rntm_l3_sup( const rntm_t* rntm )
{
	return rntm->l3_sup;
}

//
// -- rntm_t modification (internal use only) ----------------------------------
//

BLIS_INLINE void bli_rntm_set_thread_impl_only( timpl_t thread_impl, rntm_t* rntm )
{
	rntm->thread_impl = thread_impl;
}

BLIS_INLINE void bli_rntm_set_auto_factor_only( bool auto_factor, rntm_t* rntm )
{
	rntm->auto_factor = auto_factor;
}

BLIS_INLINE void bli_rntm_set_num_threads_only( dim_t nt, rntm_t* rntm )
{
	rntm->num_threads = nt;
}

BLIS_INLINE void bli_rntm_set_ways_for_only( bszid_t loop, dim_t n_ways, rntm_t* rntm )
{
	rntm->thrloop[ loop ] = n_ways;
}

BLIS_INLINE void bli_rntm_set_jc_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_NC, ways, rntm );
}
BLIS_INLINE void bli_rntm_set_pc_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_KC, 1, rntm );
}
BLIS_INLINE void bli_rntm_set_ic_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_MC, ways, rntm );
}
BLIS_INLINE void bli_rntm_set_jr_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_NR, ways, rntm );
}
BLIS_INLINE void bli_rntm_set_ir_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_MR, ways, rntm );
}
BLIS_INLINE void bli_rntm_set_pr_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_KR, ways, rntm );
}

BLIS_INLINE void bli_rntm_set_ways_only( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir, rntm_t* rntm )
{
	// Record the number of ways of parallelism per loop.
	bli_rntm_set_jc_ways_only( jc, rntm );
	bli_rntm_set_pc_ways_only(  1, rntm );
	bli_rntm_set_ic_ways_only( ic, rntm );
	bli_rntm_set_jr_ways_only( jr, rntm );
	bli_rntm_set_ir_ways_only( ir, rntm );
	bli_rntm_set_pr_ways_only(  1, rntm );
}

BLIS_INLINE void bli_rntm_clear_num_threads_only( rntm_t* rntm )
{
	bli_rntm_set_num_threads_only( 1, rntm );
}
BLIS_INLINE void bli_rntm_clear_ways_only( rntm_t* rntm )
{
	bli_rntm_set_ways_only( 1, 1, 1, 1, 1, rntm );
}

//
// -- rntm_t modification (public API) -----------------------------------------
//

BLIS_INLINE void bli_rntm_set_thread_impl( timpl_t thread_impl, rntm_t* rntm )
{
	// Set the threading implementation to use.
	bli_rntm_set_thread_impl_only( thread_impl, rntm );
}

BLIS_INLINE void bli_rntm_set_pack_a( bool pack_a, rntm_t* rntm )
{
	// Set the bool indicating whether matrix A should be packed.
	rntm->pack_a = pack_a;
}
BLIS_INLINE void bli_rntm_set_pack_b( bool pack_b, rntm_t* rntm )
{
	// Set the bool indicating whether matrix B should be packed.
	rntm->pack_b = pack_b;
}

BLIS_INLINE void bli_rntm_set_l3_sup( bool l3_sup, rntm_t* rntm )
{
	// Set the bool indicating whether level-3 sup handling is enabled.
	rntm->l3_sup = l3_sup;
}
BLIS_INLINE void bli_rntm_enable_l3_sup( rntm_t* rntm )
{
	bli_rntm_set_l3_sup( TRUE, rntm );
}
BLIS_INLINE void bli_rntm_disable_l3_sup( rntm_t* rntm )
{
	bli_rntm_set_l3_sup( FALSE, rntm );
}

//
// -- rntm_t modification (internal use only) ----------------------------------
//

BLIS_INLINE void bli_rntm_clear_thread_impl( rntm_t* rntm )
{
	bli_rntm_set_thread_impl_only( BLIS_SINGLE, rntm );
}

BLIS_INLINE void bli_rntm_clear_auto_factor( rntm_t* rntm )
{
	bli_rntm_set_auto_factor_only( FALSE, rntm );
}
BLIS_INLINE void bli_rntm_clear_pack_a( rntm_t* rntm )
{
	bli_rntm_set_pack_a( FALSE, rntm );
}
BLIS_INLINE void bli_rntm_clear_pack_b( rntm_t* rntm )
{
	bli_rntm_set_pack_b( FALSE, rntm );
}
BLIS_INLINE void bli_rntm_clear_l3_sup( rntm_t* rntm )
{
	bli_rntm_set_l3_sup( TRUE, rntm );
}

//
// -- rntm_t initialization ----------------------------------------------------
//

// NOTE: Initialization is not necessary as long the user calls at least ONE
// of the public "set" accessors, each of which guarantees that the rntm_t
// will be in a good state upon return.

#define BLIS_RNTM_INITIALIZER \
        { \
          /* .thread_impl = */ BLIS_SINGLE, \
\
          /* .auto_factor = */ FALSE, \
\
          /* .num_threads = */ 1, \
          /* .thrloop     = */ { 1, 1, 1, 1, 1, 1 }, \
          /* .pack_a      = */ FALSE, \
          /* .pack_b      = */ FALSE, \
          /* .l3_sup      = */ TRUE, \
        }  \

#if 0
//BLIS_INLINE void bli_rntm_clear( rntm_t* rntm )
{
	bli_rntm_clear_thread_impl( rntm );

	bli_rntm_clear_num_threads_only( rntm );
	bli_rntm_clear_ways_only( rntm );

	bli_rntm_clear_auto_factor( rntm );
	bli_rntm_clear_pack_a( rntm );
	bli_rntm_clear_pack_b( rntm );
	bli_rntm_clear_l3_sup( rntm );
}
#endif

//
// -- rntm_t total thread calculation ------------------------------------------
//

BLIS_INLINE dim_t bli_rntm_calc_num_threads
     (
       const rntm_t* rntm
     )
{
	dim_t n_threads;

	n_threads  = bli_rntm_ways_for( BLIS_NC, rntm );
	n_threads *= bli_rntm_ways_for( BLIS_KC, rntm );
	n_threads *= bli_rntm_ways_for( BLIS_MC, rntm );
	n_threads *= bli_rntm_ways_for( BLIS_NR, rntm );
	n_threads *= bli_rntm_ways_for( BLIS_MR, rntm );

	return n_threads;
}

//
// -- Function prototypes ------------------------------------------------------
//

rntm_t*              bli_global_rntm( void );
rntm_t*              bli_global_rntm_at_init( void );
bli_pthread_mutex_t* bli_global_rntm_mutex( void );

int bli_rntm_init( void );
int bli_rntm_finalize( void );

void bli_rntm_init_from_env( rntm_t* rntm );

BLIS_EXPORT_BLIS void bli_rntm_init_from_global( rntm_t* rntm );

BLIS_EXPORT_BLIS void bli_rntm_set_num_threads
     (
       dim_t   nt,
       rntm_t* rntm
     );

BLIS_EXPORT_BLIS void bli_rntm_set_ways
     (
       dim_t   jc,
       dim_t   pc,
       dim_t   ic,
       dim_t   jr,
       dim_t   ir,
       rntm_t* rntm
     );

void bli_rntm_sanitize
     (
       rntm_t* rntm
     );

BLIS_EXPORT_BLIS void bli_rntm_factorize
     (
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     );

void bli_rntm_factorize_sup
     (
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     );

void bli_rntm_print
     (
       const rntm_t* rntm
     );

dim_t bli_rntm_calc_num_threads_in
     (
       const bszid_t* bszid_cur,
       const rntm_t*  rntm
     );

#endif

