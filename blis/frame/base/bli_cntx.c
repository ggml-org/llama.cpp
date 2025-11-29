/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
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

BLIS_EXPORT_BLIS err_t bli_cntx_init( cntx_t* cntx )
{
	if ( cntx == NULL )
		return BLIS_NULL_POINTER;

	err_t error;

	error = bli_stack_init( sizeof( blksz_t ), 32, 32, BLIS_NUM_BLKSZS, &cntx->blkszs );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_init( sizeof( kerid_t ), 32, 32, BLIS_NUM_BLKSZS, &cntx->bmults );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_init( sizeof( func_t ), 32, 32, BLIS_NUM_UKRS, &cntx->ukrs );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_init( sizeof( func2_t ), 32, 32, BLIS_NUM_UKR2S, &cntx->ukr2s );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_init( sizeof( mbool_t ), 32, 32, BLIS_NUM_UKR_PREFS, &cntx->ukr_prefs );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_init( sizeof( void_fp ), 32, 32, BLIS_NUM_LEVEL3_OPS, &cntx->l3_sup_handlers );
	if ( error != BLIS_SUCCESS )
		return error;

	return BLIS_SUCCESS;
}

BLIS_EXPORT_BLIS err_t bli_cntx_free( cntx_t* cntx )
{
	if ( cntx == NULL )
		return BLIS_NULL_POINTER;

	err_t error;

	error = bli_stack_finalize( &cntx->blkszs );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_finalize(  &cntx->bmults );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_finalize( &cntx->ukrs );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_finalize( &cntx->ukr2s );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_finalize( &cntx->ukr_prefs );
	if ( error != BLIS_SUCCESS )
		return error;

	error = bli_stack_finalize( &cntx->l3_sup_handlers );
	if ( error != BLIS_SUCCESS )
		return error;

	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

void bli_cntx_set_blkszs( cntx_t* cntx, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default blocksizes. It should be called after
	// bli_cntx_init_<subconfig>_ref() so that the context begins with
	// default blocksizes across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_blkszs
	   (
	     cntx_t* cntx,
	     kerid_t bs0_id, blksz_t* blksz0, kerid_t bm0_id,
	     kerid_t bs1_id, blksz_t* blksz1, kerid_t bm1_id,
	     kerid_t bs2_id, blksz_t* blksz2, kerid_t bm2_id,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Initialize variable argument environment.
	va_list args;
	va_start( args, cntx );

	// Process blocksizes until we get a BLIS_VA_END.
	while ( true )
	{
		kerid_t bs_id = ( kerid_t )va_arg( args, kerid_t );

		// If we find a block size id of BLIS_VA_END, then we are done.
		if ( bs_id == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the kerid_t of the blocksize we're about to process (already done),
		// - the address of the blksz_t object,
		// - the kerid_t of the multiple we need to associate with
		//   the blksz_t object.
		blksz_t* blksz = ( blksz_t* )va_arg( args, blksz_t* );
		kerid_t  bm_id = ( kerid_t  )va_arg( args, kerid_t );

		// Copy the blksz_t object contents into the appropriate
		// location within the context's blksz_t array. Do the same
		// for the blocksize multiple id.
		bli_cntx_set_blksz( bs_id, blksz, bm_id, cntx );
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_ukrs( cntx_t* cntx , ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default microkernels. It should be called after
	// bli_cntx_init_<subconfig>_ref() so that the context begins with
	// default microkernels across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_ukrs
	   (
	     cntx_t* cntx,
	     kerid_t ukr0_id, num_t dt0, void_fp ukr0_fp,
	     kerid_t ukr1_id, num_t dt1, void_fp ukr1_fp,
	     kerid_t ukr2_id, num_t dt2, void_fp ukr2_fp,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Initialize variable argument environment.
	va_list   args;
	va_start( args, cntx );

	// Process ukernels until BLIS_VA_END is reached.
	while ( true )
	{
		kerid_t ukr_id = ( kerid_t )va_arg( args, kerid_t );

		// If we find a ukernel id of BLIS_VA_END, then we are done.
		if ( ukr_id == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the kerid_t of the kernel we're about to process (already done),
		// - the datatype of the kernel, and
		// - the kernel function pointer
		num_t   ukr_dt = ( num_t   )va_arg( args, num_t   );
		void_fp ukr_fp = ( void_fp )va_arg( args, void_fp );

		// Store the ukernel function pointer into the context.
		bli_cntx_set_ukr_dt( ukr_fp, ukr_dt, ukr_id, cntx );
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_ukr2s( cntx_t* cntx , ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default microkernels. It should be called after
	// bli_cntx_init_<subconfig>_ref() so that the context begins with
	// default microkernels across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_ukr2s
	   (
	     cntx_t* cntx,
	     kerid_t ukr0_id, num_t dt1_0, num_t dt2_0, void_fp ukr0_fp,
	     kerid_t ukr1_id, num_t dt1_1, num_t dt2_1, void_fp ukr1_fp,
	     kerid_t ukr2_id, num_t dt1_2, num_t dt2_2, void_fp ukr2_fp,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Initialize variable argument environment.
	va_list   args;
	va_start( args, cntx );

	// Process ukernels until BLIS_VA_END is reached.
	while ( true )
	{
		kerid_t ukr_id = ( kerid_t )va_arg( args, kerid_t );

		// If we find a ukernel id of BLIS_VA_END, then we are done.
		if ( ukr_id == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the kerid_t of the kernel we're about to process (already done),
		// - the datatype of the kernel, and
		// - the kernel function pointer
		num_t   ukr_dt1 = ( num_t   )va_arg( args, num_t   );
		num_t   ukr_dt2 = ( num_t   )va_arg( args, num_t   );
		void_fp ukr_fp  = ( void_fp )va_arg( args, void_fp );

		// Store the ukernel function pointer into the context.
		bli_cntx_set_ukr2_dt( ukr_fp, ukr_dt1, ukr_dt2, ukr_id, cntx );
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_ukr_prefs( cntx_t* cntx , ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default microkernel preferences. It should be called after
	// bli_cntx_init_<subconfig>_ref() so that the context begins with
	// default preferences across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_ukr_prefs
	   (
	     cntx_t* cntx,
	     kerid_t ukr_pref0_id, num_t dt0, bool ukr_pref0,
	     kerid_t ukr_pref1_id, num_t dt1, bool ukr_pref1,
	     kerid_t ukr_pref2_id, num_t dt2, bool ukr_pref2,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Initialize variable argument environment.
	va_list   args;
	va_start( args, cntx );

	// Process ukernel preferences until BLIS_VA_END is reached.
	while ( true )
	{
		kerid_t ukr_pref_id = ( kerid_t )va_arg( args, kerid_t );

		// If we find a ukernel pref id of BLIS_VA_END, then we are done.
		if ( ukr_pref_id == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the kerid_t of the kernel we're about to process (already done),
		// - the datatype of the kernel, and
		// - the kernel function pointer
		num_t ukr_pref_dt = ( num_t )va_arg( args, num_t );
		bool  ukr_pref    = ( bool  )va_arg( args, int );

		// Store the ukernel preference value into the context.
		bli_cntx_set_ukr_pref_dt( ukr_pref, ukr_pref_dt, ukr_pref_id, cntx );
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_l3_sup_handlers( cntx_t* cntx, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default level-3 operation handler for small/unpacked matrices. It
	// should be called after bli_cntx_init_<subconfig>_ref() so that the
	// context begins with default sup handlers across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_l3_sup_handlers
	   (
	     cntx_t* cntx
	     opid_t  op0_id, void_fp handler0_fp,
	     opid_t  op1_id, void_fp handler1_fp,
	     opid_t  op2_id, void_fp handler2_fp,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Initialize variable argument environment.
	va_list   args;
	va_start( args, cntx );

	// Process sup handlers until BLIS_VA_END is reached.
	while ( true )
	{
		kerid_t op_id = ( kerid_t )va_arg( args, kerid_t );

		// If we find an operation id of BLIS_VA_END, then we are done.
		if ( op_id == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the opid_t of the operation we're about to process,
		// - the sup handler function pointer
		      void_fp op_fp = ( void_fp )va_arg( args, void_fp );

		if ( op_id >= BLIS_NUM_LEVEL3_OPS )
			bli_abort();

		// Store the sup handler function pointer into the slot for the
		// specified operation id.
		void_fp* l3_sup_handler;
		bli_stack_get( op_id, ( void** )&l3_sup_handler, &cntx->l3_sup_handlers );
		*l3_sup_handler = op_fp;
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

err_t bli_cntx_register_blksz( kerid_t* bs_id, const blksz_t* blksz, kerid_t bmult_id, cntx_t* cntx )
{
	siz_t id_blksz;
	err_t error = bli_stack_push( &id_blksz, &cntx->blkszs );
	if ( error != BLIS_SUCCESS )
		return error;

	siz_t id_bmult;
	error = bli_stack_push( &id_bmult, &cntx->bmults );
	if ( error != BLIS_SUCCESS )
		return error;

	if ( id_blksz != id_bmult )
		return BLIS_INVALID_UKR_ID;

	*bs_id = id_blksz;

	if ( blksz )
	{
		return bli_cntx_set_blksz( id_blksz, blksz, bmult_id, cntx );
	}
	else
	{
		return BLIS_SUCCESS;
	}
}

err_t bli_cntx_register_ukr( kerid_t* ukr_id, const func_t* ukr, cntx_t* cntx )
{
	siz_t new_ukr_id;
	err_t error = bli_stack_push( &new_ukr_id, &cntx->ukrs );
	if ( error != BLIS_SUCCESS )
		return error;

	*ukr_id = new_ukr_id;

	if ( ukr )
	{
		return bli_cntx_set_ukr( *ukr_id, ukr, cntx );
	}
	else
	{
		return BLIS_SUCCESS;
	}
}

err_t bli_cntx_register_ukr2( kerid_t* ukr_id, const func2_t* ukr, cntx_t* cntx )
{
	siz_t new_ukr_id;
	err_t error = bli_stack_push( &new_ukr_id, &cntx->ukr2s );
	if ( error != BLIS_SUCCESS )
		return error;

	*ukr_id = new_ukr_id;

	if ( ukr )
	{
		return bli_cntx_set_ukr2( *ukr_id, ukr, cntx );
	}
	else
	{
		return BLIS_SUCCESS;
	}
}

err_t bli_cntx_register_ukr_pref( kerid_t* ukr_pref_id, const mbool_t* ukr_pref, cntx_t* cntx )
{
	siz_t new_ukr_pref_id;
	err_t error = bli_stack_push( &new_ukr_pref_id, &cntx->ukr_prefs );
	if ( error != BLIS_SUCCESS )
		return error;

	*ukr_pref_id = new_ukr_pref_id;

	if ( ukr_pref )
	{
		return bli_cntx_set_ukr_pref( *ukr_pref_id, ukr_pref, cntx );
	}
	else
	{
		return BLIS_SUCCESS;
	}
}


// -----------------------------------------------------------------------------

void bli_cntx_print( const cntx_t* cntx )
{
	dim_t i;

	// Print the values stored in the blksz_t objects.
	printf( "                               s                d                c                z\n" );

	for ( i = 0; i < BLIS_NUM_BLKSZS; ++i )
	{
		printf( "blksz/mult %2lu:  %13lu/%2lu %13lu/%2lu %13lu/%2lu %13lu/%2lu\n",
		         ( unsigned long )i,
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_FLOAT,    i, cntx ),
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_DOUBLE,   i, cntx ),
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_SCOMPLEX, i, cntx ),
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_DCOMPLEX, i, cntx )
		      );
	}

	for ( i = 0; i < BLIS_NUM_UKRS; ++i )
	{
		const func_t* ukr = bli_cntx_get_ukrs( i, cntx );

		printf( "ukr %2lu:  %16p %16p %16p %16p\n",
		        ( unsigned long )i,
		        bli_func_get_dt( BLIS_FLOAT,    ukr ),
		        bli_func_get_dt( BLIS_DOUBLE,   ukr ),
		        bli_func_get_dt( BLIS_SCOMPLEX, ukr ),
		        bli_func_get_dt( BLIS_DCOMPLEX, ukr )
		      );
	}

	for ( i = 0; i < BLIS_NUM_UKR_PREFS; ++i )
	{
		const mbool_t* ukr_pref = bli_cntx_get_ukr_prefs( i, cntx );

		printf( "ukr pref %2lu:  %d %d %d %d\n",
		        ( unsigned long )i,
		        bli_mbool_get_dt( BLIS_FLOAT,    ukr_pref ),
		        bli_mbool_get_dt( BLIS_DOUBLE,   ukr_pref ),
		        bli_mbool_get_dt( BLIS_SCOMPLEX, ukr_pref ),
		        bli_mbool_get_dt( BLIS_DCOMPLEX, ukr_pref )
		      );
	}
}

