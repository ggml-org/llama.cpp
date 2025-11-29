/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2020, Advanced Micro Devices, Inc.

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

// This array tracks whether a particular operation is implemented for each of
// the induced methods. This array is meant to be read-only.
static const bool bli_l3_ind_oper_impl[BLIS_NUM_IND_METHODS][BLIS_NUM_LEVEL3_OPS] =
{
        /*   gemm  gemmt  hemm  herk  her2k  symm  syrk  syr2k  trmm3  trmm  trsm  */
/* 1m   */ { TRUE, TRUE,  TRUE, TRUE, TRUE,  TRUE, TRUE, TRUE,  TRUE,  TRUE, TRUE  },
/* nat  */ { TRUE, TRUE,  TRUE, TRUE, TRUE,  TRUE, TRUE, TRUE,  TRUE,  TRUE, TRUE  }
};

//
// NOTE: "2" is used instead of BLIS_NUM_FP_TYPES/2.
//
// BLIS provides APIs to modify this state during runtime. So, it's possible for one
// application thread to modify the state before another starts the corresponding
// BLIS operation. This is solved by making the induced method status array local to
// threads.

static BLIS_THREAD_LOCAL
bool bli_l3_ind_oper_st[BLIS_NUM_IND_METHODS][BLIS_NUM_LEVEL3_OPS][2] =
{
        /*   gemm           gemmt          hemm           herk           her2k          symm
             syrk           syr2k          trmm3          trmm           trsm  */
        /*    c     z    */
/* 1m   */ { {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE},
             {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}, {FALSE,FALSE}  },
/* nat  */ { {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},
             {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE},   {TRUE,TRUE}    },
};

// A mutex to allow synchronous access to the bli_l3_ind_oper_st array.
#ifdef BLIS_DISABLE_TLS
static bli_pthread_mutex_t oper_st_mutex = BLIS_PTHREAD_MUTEX_INITIALIZER;
#endif

// -----------------------------------------------------------------------------

#undef  GENFUNC
#define GENFUNC( opname, optype ) \
\
ind_t PASTEMAC(opname,ind_find_avail)( num_t dt ) \
{ \
	return bli_l3_ind_oper_find_avail( optype, dt ); \
}
//bool PASTEMAC(opname,ind_has_avail)( num_t dt )
//{
//	return bli_ind_oper_has_avail( optype, dt );
//}

GENFUNC( gemm, BLIS_GEMM )
GENFUNC( gemmt, BLIS_GEMMT )
GENFUNC( hemm, BLIS_HEMM )
GENFUNC( symm, BLIS_SYMM )
GENFUNC( trmm3, BLIS_TRMM3 )
GENFUNC( trmm, BLIS_TRMM )
GENFUNC( trsm, BLIS_TRSM )

// -----------------------------------------------------------------------------

#if 0
bool bli_l3_ind_oper_is_avail( opid_t oper, ind_t method, num_t dt )
{
	bool enabled;
	bool stat;

	// If the datatype is real, it is never available.
	if ( !bli_is_complex( dt ) ) return FALSE;

	enabled = bli_l3_ind_oper_is_impl( oper, method );
	stat    = bli_l3_ind_oper_get_enable( oper, method, dt );

	return ( enabled == TRUE && stat == TRUE );
}
#endif

// -----------------------------------------------------------------------------

ind_t bli_l3_ind_oper_find_avail( opid_t oper, num_t dt )
{
	bli_init_once();

	ind_t im;

	// If the datatype is real, return native execution.
	if ( !bli_is_complex( dt ) ) return BLIS_NAT;

	// If the operation is not level-3, return native execution.
	if ( !bli_opid_is_level3( oper ) ) return BLIS_NAT;

	// Iterate over all induced methods and search for the first one
	// that is available (ie: both implemented and enabled) for the
	// current operation and datatype.
	for ( im = 0; im < BLIS_NUM_IND_METHODS; ++im )
	{
		bool enabled = bli_l3_ind_oper_is_impl( oper, im );
		bool stat    = bli_l3_ind_oper_get_enable( oper, im, dt );

		if ( enabled == TRUE &&
		     stat    == TRUE ) return im;
	}

	// This return statement should never execute since the native index
	// should be found even if all induced methods are unavailable. We
	// include it simply to avoid a compiler warning.
	return BLIS_NAT;
}

// -----------------------------------------------------------------------------

void bli_l3_ind_set_enable_dt( ind_t method, num_t dt, bool status )
{
	opid_t iop;

	if ( !bli_is_complex( dt ) ) return;

	// Iterate over all level-3 operation ids.
	for ( iop = 0; iop < BLIS_NUM_LEVEL3_OPS; ++iop )
	{
		bli_l3_ind_oper_set_enable( iop, method, dt, status );
	}
}

// -----------------------------------------------------------------------------

void bli_l3_ind_oper_enable_only( opid_t oper, ind_t method, num_t dt )
{
	ind_t im;

	if ( !bli_is_complex( dt ) ) return;
	if ( !bli_opid_is_level3( oper ) ) return;

	for ( im = 0; im < BLIS_NUM_IND_METHODS; ++im )
	{
		// Native execution should always stay enabled.
		if ( im == BLIS_NAT ) continue;

		// When we come upon the requested method, enable it for the given
		// operation and datatype. Otherwise, disable it.
		if ( im == method )
			bli_l3_ind_oper_set_enable( oper, im, dt, TRUE );
		else
			bli_l3_ind_oper_set_enable( oper, im, dt, FALSE );
	}
}

void bli_l3_ind_oper_set_enable_all( opid_t oper, num_t dt, bool status )
{
	ind_t im;

	if ( !bli_is_complex( dt ) ) return;
	if ( !bli_opid_is_level3( oper ) ) return;

	for ( im = 0; im < BLIS_NUM_IND_METHODS; ++im )
	{
		// Native execution should always stay enabled.
		if ( im != BLIS_NAT )
			bli_l3_ind_oper_set_enable( oper, im, dt, status );
	}
}

// -----------------------------------------------------------------------------

void bli_l3_ind_oper_set_enable( opid_t oper, ind_t method, num_t dt, bool status )
{
	num_t idt;

	if ( !bli_is_complex( dt ) ) return;
	if ( !bli_opid_is_level3( oper ) ) return;

	// BLIS currently implements herk/her2k/syrk/syr2k in terms of the user-
	// level gemmt (expert) API, and so those operations choose to execute
	// 1m (or not) based on the induced method enablement status of gemmt.
	// In other words, changing the enablement status of those operations
	// would have no effect. Therefore, we redirect queries/accesses to those
	// operations' induced method enablement statuses to that of gemmt.
	if ( method != BLIS_NAT && ( oper == BLIS_HERK  ||
	                             oper == BLIS_HER2K ||
	                             oper == BLIS_SYRK  ||
	                             oper == BLIS_SYR2K ) )
		oper = BLIS_GEMMT;

	// Disallow changing status of native execution.
	if ( method == BLIS_NAT ) return;

	idt = bli_ind_map_cdt_to_index( dt );

	// If TLS is disabled, we need to use a mutex to protect the status array
	// since it will be shared with all application threads.
	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_lock( &oper_st_mutex );
	#endif

	// BEGIN CRITICAL SECTION
	{
		bli_l3_ind_oper_st[ method ][ oper ][ idt ] = status;
	}
	// END CRITICAL SECTION

	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_unlock( &oper_st_mutex );
	#endif
}

bool bli_l3_ind_oper_get_enable( opid_t oper, ind_t method, num_t dt )
{
	num_t idt = bli_ind_map_cdt_to_index( dt );
	bool  r_val;

	// BLIS currently implements herk/her2k/syrk/syr2k in terms of the user-
	// level gemmt (expert) API, and so those operations choose to execute
	// 1m (or not) based on the induced method enablement status of gemmt.
	// In other words, changing the enablement status of those operations
	// would have no effect. Therefore, we redirect queries/accesses to those
	// operations' induced method enablement statuses to that of gemmt.
	if ( method != BLIS_NAT && ( oper == BLIS_HERK  ||
	                             oper == BLIS_HER2K ||
	                             oper == BLIS_SYRK  ||
	                             oper == BLIS_SYR2K ) )
		oper = BLIS_GEMMT;

	{
		r_val = bli_l3_ind_oper_st[ method ][ oper ][ idt ];
	}

	return r_val;
}

// -----------------------------------------------------------------------------

bool bli_l3_ind_oper_is_impl( opid_t oper, ind_t method )
{
	return bli_l3_ind_oper_impl[ method ][ oper ];
}
