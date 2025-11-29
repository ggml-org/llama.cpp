/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018-2020, Advanced Micro Devices, Inc.
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

// The array of cntx_t* pointers to cache modified contexts used by induced
// methods.
static cntx_t* gks[ BLIS_NUM_ARCHS ];

// Define a function pointer type for context initialization functions.
typedef void (*cntx_init_ft)( cntx_t* cntx );

// The array of function pointers holding the registered context initialization
// functions for reference kernels.
static cntx_init_ft cntx_ref_init[ BLIS_NUM_ARCHS ];

// Cached copies of the pointers to the native context for the
// active subconfiguration. When BLIS_ENABLE_GKS_CACHING is enabled, these
// pointers will be set once and then reused to fulfill subsequent context
// queries.
static cntx_t* cached_cntx = NULL;

// -----------------------------------------------------------------------------

int bli_gks_init( void )
{
	// NOTE: This function is called once by ONLY ONE application thread per
	// library init/finalize cycle (see bli_init.c). Thus, a mutex is not
	// needed to protect the data initialization.

	// Initialize the internal data structure we use to track registered
	// contexts.
	bli_gks_init_index();

	// Register a context for each architecture that was #define'd in
	// bli_config.h.

	#undef GENTCONF
	#define GENTCONF( CONFIG, config ) \
	\
	bli_gks_register_cntx( PASTECH(BLIS_ARCH_,CONFIG), \
	                       PASTEMAC(cntx_init_,config), \
	                       PASTEMAC(cntx_init_,config,_ref) );

	INSERT_GENTCONF

#ifdef BLIS_ENABLE_GKS_CACHING
	// Deep-query and cache the native and induced method contexts so they are
	// ready to go when needed (by BLIS or the application). Notice that we use
	// the _noinit() APIs, which skip their internal calls to bli_init_once().
	// The reasons: (1) Skipping that call is necessary to prevent an infinite
	// loop since the current function, bli_gks_init(), is called from within
	// bli_init_once(); and (2) we can guarantee that the gks has been
	// initialized given that bli_gks_init() is about to return.
	cached_cntx = ( cntx_t* )bli_gks_query_cntx_noinit();
#endif

	return 0;
}

// -----------------------------------------------------------------------------

int bli_gks_finalize( void )
{
	arch_t id;

	// BEGIN CRITICAL SECTION
	// NOTE: This critical section is implicit. We assume this function is only
	// called from within the critical section within bli_finalize().
	{
		// Iterate over the architectures in the gks array.
		for ( id = 0; id < BLIS_NUM_ARCHS; ++id )
		{
			cntx_t* gks_id = gks[ id ];

			// Only consider context arrays for architectures that were allocated
			// in the first place.
			if ( gks_id != NULL )
			{
				#ifdef BLIS_ENABLE_MEM_TRACING
				printf( "bli_gks_finalize(): cntx for ind_t %d: ", ( int )id );
				#endif

				bli_cntx_free( gks_id );
				bli_free_intl( gks_id );
			}
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_GKS_CACHING
	// Clear the cached pointers to the native and induced contexts.
	cached_cntx = NULL;
#endif

	return 0;
}

// -----------------------------------------------------------------------------

void bli_gks_init_index( void )
{
	// This function is called by bli_gks_init(). It simply initializes all
	// architecture id elements of the internal arrays to NULL.

	const size_t gks_size = sizeof( cntx_t* ) * BLIS_NUM_ARCHS;
	const size_t fpa_size = sizeof( void_fp ) * BLIS_NUM_ARCHS;

	// Set every entry in gks and context init function pointer arrays to
	// zero/NULL. This is done so that later on we know which ones were
	// allocated.
	memset( gks,           0, gks_size );
	memset( cntx_ref_init, 0, fpa_size );
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_lookup_id
     (
       arch_t id
     )
{
	// Return the address of the array of context pointers for a given
	// architecture id. This function is only used for sanity check purposes
	// to ensure that the underlying data structures for a particular id are
	// initialized.

	// Sanity check: verify that the arch_t id is valid.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_valid_arch_id( id );
		bli_check_error_code( e_val );
	}

	// Index into the array of context pointers for the given architecture id.
	return gks[ id ];
}

// -----------------------------------------------------------------------------

void bli_gks_register_cntx
     (
       arch_t  id,
       void_fp nat_fp,
       void_fp ref_fp
     )
{
	err_t e_val;

	// This function is called by bli_gks_init() for each architecture that
	// will be supported by BLIS. It takes an architecture id and two
	// function pointers, one to a function that initializes a native context
	// (supplied by the kernel developer), and one to a function that initializes
	// a reference context (with function pointers specific to the architecture
	// associated with id). The latter function is automatically
	// generated by the framework. Unlike with native contexts, we don't
	// ever store reference contexts. For this reason, we
	// can get away with only storing the pointers to the initialization
	// functions for this type of context, which we can then
	// call at a later time when the reference context is needed.

	// Sanity check: verify that the arch_t id is valid.
	if ( bli_error_checking_is_enabled() )
	{
		e_val = bli_check_valid_arch_id( id );
		bli_check_error_code( e_val );
	}

	cntx_init_ft f = nat_fp;

	// First, store the function pointer to the context initialization
	// function for reference kernels. This
	// will be used whenever we need to obtain reference kernels.
	cntx_ref_init[ id ] = ref_fp;

	// If the the context array pointer isn't NULL, then it means the given
	// architecture id has already registered (and the underlying memory
	// allocations and context initializations have already been performed).
	// This is really just a safety feature to prevent memory leaks; this
	// early return should never occur, because the caller should never try
	// to register with an architecture id that has already been registered.
	if ( gks[ id ] != NULL ) return;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_gks_register_cntx(): " );
	#endif

	// Allocate memory for a single context and store the address at
	// the element in the gks[ id ] array that is reserved for native
	// execution.
	cntx_t* gks_id = bli_calloc_intl( sizeof( cntx_t ), &e_val );
	gks[ id ] = gks_id;

	// The context structure is initialied in bli_cntx_init_<config>_ref

	// Call the context initialization function on the element of the newly
	// allocated array corresponding to native execution.
	f( gks_id );

	// Verify that cache blocksizes are whole multiples of register blocksizes.
	// Specifically, verify that:
	//   - MC is a whole multiple of MR.
	//   - NC is a whole multiple of NR.
	//   - KC is a whole multiple of KR.
	// These constraints are enforced because it makes it easier to handle diagonals
	// in the macro-kernel implementations. Additionally, we optionally verify that:
	//   - MC is a whole multiple of NR.
	//   - NC is a whole multiple of MR.
	// These latter constraints, guarded by #ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
	// below, are only enforced when we wish to be able to handle the trsm right-
	// side case handling that swaps A and B, so that B is the triangular matrix,
	// with NR blocking used to pack A and MR blocking used to pack B, with the
	// arguments to the gemmtrsm microkernel swapped at the last minute, as the
	// kernel is called.

	const mbool_t* row_pref = bli_cntx_get_ukr_prefs( BLIS_GEMM_UKR_ROW_PREF, gks_id );
	const blksz_t* mc       = bli_cntx_get_blksz( BLIS_MC, gks_id );
	const blksz_t* nc       = bli_cntx_get_blksz( BLIS_NC, gks_id );
	const blksz_t* kc       = bli_cntx_get_blksz( BLIS_KC, gks_id );
	const blksz_t* mr       = bli_cntx_get_blksz( BLIS_MR, gks_id );
	const blksz_t* nr       = bli_cntx_get_blksz( BLIS_NR, gks_id );
	const blksz_t* kr       = bli_cntx_get_blksz( BLIS_KR, gks_id );

	e_val = bli_check_valid_mc_mod_mult( mc, mr ); bli_check_error_code( e_val );
	e_val = bli_check_valid_nc_mod_mult( nc, nr ); bli_check_error_code( e_val );
	e_val = bli_check_valid_kc_mod_mult( kc, kr ); bli_check_error_code( e_val );
#ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
	e_val = bli_check_valid_mc_mod_mult( mc, nr ); bli_check_error_code( e_val );
	e_val = bli_check_valid_nc_mod_mult( nc, mr ); bli_check_error_code( e_val );
#endif

	e_val = bli_check_valid_mr_even( mr, row_pref ); bli_check_error_code( e_val );
	e_val = bli_check_valid_nr_even( nr, row_pref ); bli_check_error_code( e_val );

	// Verify that the register blocksizes in the context are sufficiently large
	// relative to the maximum stack buffer size defined at configure-time.
	e_val = bli_check_sufficient_stack_buf_size( gks_id );
	bli_check_error_code( e_val );
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_query_cntx( void )
{
	bli_init_once();

#ifdef BLIS_ENABLE_GKS_CACHING

	// Return a pointer to the context for native execution that was deep-
	// queried and cached at the end of bli_gks_init().
	return cached_cntx;

#else

	// Deep-query and return the address of a context for native execution.
	return bli_gks_query_cntx_impl();

#endif
}

const cntx_t* bli_gks_query_cntx_noinit( void )
{
	// NOTE: This function purposefully avoids calling bli_init_once() so that
	// it is safe to call during inititalization.

	return bli_gks_query_cntx_impl();
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_query_cntx_impl( void )
{
	// Return the address of the native context for the architecture id
	// corresponding to the current hardware, as determined by
	// bli_arch_query_id().

	// Query the architecture id.
	arch_t id = bli_arch_query_id();

	// Use the architecture id to look up a pointer to its context.
	const cntx_t* cntx = bli_gks_lookup_id( id );

	return cntx;
}

// -----------------------------------------------------------------------------

void bli_gks_init_ref_cntx
    (
      cntx_t* cntx
    )
{
	// Query the architecture id.
	arch_t id = bli_arch_query_id();

	// Sanity check: verify that the arch_t id is valid.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_valid_arch_id( id );
		bli_check_error_code( e_val );
	}

	// Obtain the function pointer to the context initialization function for
	// reference kernels.
	cntx_init_ft f = cntx_ref_init[ id ];

	// Initialize the caller's context with reference kernels and related values.
	f( cntx );
}

// -----------------------------------------------------------------------------

bool bli_gks_cntx_ukr_is_ref
     (
             num_t   dt,
             ukr_t   ukr_id,
       const cntx_t* cntx
     )
{
	cntx_t ref_cntx;

	// Initialize a context with reference kernels for the arch_t id queried
	// via bli_arch_query_id().
	bli_gks_init_ref_cntx( &ref_cntx );

	// Query each context for the micro-kernel function pointer for the
	// specified datatype.
	void_fp ref_fp = bli_cntx_get_ukr_dt( dt, ukr_id, &ref_cntx );
	void_fp fp     = bli_cntx_get_ukr_dt( dt, ukr_id, cntx );

    bli_cntx_free( &ref_cntx );

	// Return the result.
	return fp == ref_fp;
}

bool bli_gks_cntx_ukr2_is_ref
     (
             num_t   dt1,
             num_t   dt2,
             ukr_t   ukr_id,
       const cntx_t* cntx
     )
{
	cntx_t ref_cntx;

	// Initialize a context with reference kernels for the arch_t id queried
	// via bli_arch_query_id().
	bli_gks_init_ref_cntx( &ref_cntx );

	// Query each context for the micro-kernel function pointer for the
	// specified datatype.
	void_fp ref_fp = bli_cntx_get_ukr2_dt( dt1, dt2, ukr_id, &ref_cntx );
	void_fp fp     = bli_cntx_get_ukr2_dt( dt1, dt2, ukr_id, cntx );

    bli_cntx_free( &ref_cntx );

	// Return the result.
	return fp == ref_fp;
}

//
// -- level-3 micro-kernel implementation strings ------------------------------
//

static const char* bli_gks_l3_ukr_impl_str[BLIS_NUM_UKR_IMPL_TYPES] =
{
	"refrnce",
	"virtual",
	"optimzd",
	"notappl",
};

// -----------------------------------------------------------------------------

const char* bli_gks_l3_ukr_impl_string( ukr_t ukr, ind_t method, num_t dt )
{
	kimpl_t ki;

	// Query the context for the current induced method and datatype, and
	// then query the ukernel function pointer for the given datatype from
	// that context.
	const cntx_t* cntx = bli_gks_query_cntx();
	void_fp fp         = bli_cntx_get_ukr_dt( dt, ukr, cntx );

	// Check whether the ukernel function pointer is NULL for the given
	// datatype. If it is NULL, return the string for not applicable.
	// Otherwise, query the ukernel implementation type using the method
	// provided and return the associated string.
	if ( fp == NULL )
		ki = BLIS_NOTAPPLIC_UKERNEL;
	else
		ki = bli_gks_l3_ukr_impl_type( ukr, method, dt );

	return bli_gks_l3_ukr_impl_str[ ki ];
}

#if 0
char* bli_gks_l3_ukr_avail_impl_string( ukr_t ukr, num_t dt )
{
	opid_t  oper;
	ind_t   method;
	kimpl_t ki;

	// We need to decide which operation we will use to query the
	// current available induced method. If the ukr type given is
	// BLIS_GEMM_UKR, we use gemm. Otherwise, we use trsm (since
	// the four other defined ukr types are trsm-related).
	if ( ukr == BLIS_GEMM_UKR ) oper = BLIS_GEMM;
	else                        oper = BLIS_TRSM;

	// Query the current available induced method using the
	// chosen operation id type.
	method = bli_l3_ind_oper_find_avail( oper, dt );

	// Query the ukernel implementation type using the current
	// available method.
	ki = bli_gks_l3_ukr_impl_type( ukr, method, dt );

	return bli_ukr_impl_str[ ki ];
}
#endif

kimpl_t bli_gks_l3_ukr_impl_type( ukr_t ukr, ind_t method, num_t dt )
{
	// If the current available induced method is not native, it
	// must be virtual.
	if ( method != BLIS_NAT ) return BLIS_VIRTUAL_UKERNEL;
	else
	{
		// If the current available induced method for the gemm
		// operation is native, then it might be reference or
		// optimized. To determine which, we compare the
		// datatype-specific function pointer within the ukrs
		// object corresponding to the current available induced
		// method to the typed function pointer within the known
		// reference ukrs object.

		// Query the native context from the gks.
		const cntx_t* cntx = bli_gks_query_cntx();

		if ( bli_gks_cntx_ukr_is_ref( dt, ukr, cntx ) )
			return BLIS_REFERENCE_UKERNEL;
		else
			return BLIS_OPTIMIZED_UKERNEL;
	}
}

//
// -- microkernel and block size registration ----------------------------------
//

err_t bli_gks_register_blksz( kerid_t* bs_id )
{
	kerid_t id = 0;
	kerid_t next_id;
	cntx_t* cntx;
	err_t err;

	#undef GENTCONF
	#define GENTCONF( CONFIG, config ) \
	\
	cntx = ( cntx_t* )bli_gks_lookup_id( PASTECH(BLIS_ARCH_,CONFIG) ); \
	err = bli_cntx_register_blksz( &next_id, NULL, 0, cntx ); \
	if ( err != BLIS_SUCCESS ) \
	{ \
		*bs_id = 0; \
		return err; \
	} \
	if ( id != 0 && id != next_id ) \
	{ \
		*bs_id = 0; \
		return BLIS_INVALID_UKR_ID; \
	} \
	id = next_id;

	INSERT_GENTCONF

	*bs_id = id;

	return BLIS_SUCCESS;
}

err_t bli_gks_register_ukr( kerid_t* ukr_id )
{
	kerid_t id = 0;
	kerid_t next_id;
	cntx_t* cntx;
	err_t err;

	#undef GENTCONF
	#define GENTCONF( CONFIG, config ) \
	\
	cntx = ( cntx_t* )bli_gks_lookup_id( PASTECH(BLIS_ARCH_,CONFIG) ); \
	err = bli_cntx_register_ukr( &next_id, NULL, cntx ); \
	if ( err != BLIS_SUCCESS ) \
	{ \
		*ukr_id = 0; \
		return err; \
	} \
	if ( id != 0 && id != next_id ) \
	{ \
		*ukr_id = 0; \
		return BLIS_INVALID_UKR_ID; \
	} \
	id = next_id;

	INSERT_GENTCONF

	*ukr_id = id;

	return BLIS_SUCCESS;
}

err_t bli_gks_register_ukr2( kerid_t* ukr_id )
{
	kerid_t id = 0;
	kerid_t next_id;
	cntx_t* cntx;
	err_t err;

	#undef GENTCONF
	#define GENTCONF( CONFIG, config ) \
	\
	cntx = ( cntx_t* )bli_gks_lookup_id( PASTECH(BLIS_ARCH_,CONFIG) ); \
	err = bli_cntx_register_ukr2( &next_id, NULL, cntx ); \
	if ( err != BLIS_SUCCESS ) \
	{ \
		*ukr_id = 0; \
		return err; \
	} \
	if ( id != 0 && id != next_id ) \
	{ \
		*ukr_id = 0; \
		return BLIS_INVALID_UKR_ID; \
	} \
	id = next_id;

	INSERT_GENTCONF

	*ukr_id = id;

	return BLIS_SUCCESS;
}

err_t bli_gks_register_ukr_pref( kerid_t* ukr_pref_id )
{
	kerid_t id = 0;
	kerid_t next_id;
	cntx_t* cntx;
	err_t err;

	#undef GENTCONF
	#define GENTCONF( CONFIG, config ) \
	\
	cntx = ( cntx_t* )bli_gks_lookup_id( PASTECH(BLIS_ARCH_,CONFIG) ); \
	err = bli_cntx_register_ukr_pref( &next_id, NULL, cntx ); \
	if ( err != BLIS_SUCCESS ) \
	{ \
		*ukr_pref_id = 0; \
		return err; \
	} \
	if ( id != 0 && id != next_id ) \
	{ \
		*ukr_pref_id = 0; \
		return BLIS_INVALID_UKR_ID; \
	} \
	id = next_id;

	INSERT_GENTCONF

	*ukr_pref_id = id;

	return BLIS_SUCCESS;
}

