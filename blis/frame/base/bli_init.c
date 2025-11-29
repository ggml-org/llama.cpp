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

// -----------------------------------------------------------------------------

void bli_init( void )
{
	bli_init_once();
}

void bli_finalize( void )
{
	bli_finalize_once();
}

// -----------------------------------------------------------------------------

void bli_init_auto( void )
{
	bli_init_once();
}

void bli_finalize_auto( void )
{
	// The _auto() functions are used when initializing the BLAS compatibility
	// layer. It would not make much sense to automatically initialize and
	// finalize for every BLAS routine call; therefore, we remain initialized
	// unless and until the application explicitly calls bli_finalize().
}

// -----------------------------------------------------------------------------

void bli_init_once( void )
{
	bli_init_apis();
}

void bli_finalize_once( void )
{
	bli_finalize_apis();
}

// -----------------------------------------------------------------------------

static bli_pthread_switch_t gks_g_state    = BLIS_PTHREAD_SWITCH_INIT;
static BLIS_THREAD_LOCAL
       bli_pthread_switch_t ind_l_state    = BLIS_PTHREAD_SWITCH_INIT;
static bli_pthread_switch_t thread_g_state = BLIS_PTHREAD_SWITCH_INIT;
static BLIS_THREAD_LOCAL
       bli_pthread_switch_t rntm_l_state   = BLIS_PTHREAD_SWITCH_INIT;
static bli_pthread_switch_t memsys_g_state = BLIS_PTHREAD_SWITCH_INIT;

int bli_init_apis( void )
{
	// Initialize various sub-APIs.
	bli_pthread_switch_on( &gks_g_state,    bli_gks_init );
	bli_pthread_switch_on( &ind_l_state,    bli_ind_init );
	bli_pthread_switch_on( &thread_g_state, bli_thread_init );
	bli_pthread_switch_on( &rntm_l_state,   bli_rntm_init );
	bli_pthread_switch_on( &memsys_g_state, bli_memsys_init );

	return 0;
}

int bli_finalize_apis( void )
{
	// Finalize various sub-APIs.
	bli_pthread_switch_off( &memsys_g_state, bli_memsys_finalize );
	bli_pthread_switch_off( &rntm_l_state,   bli_rntm_finalize );
	bli_pthread_switch_off( &thread_g_state, bli_thread_finalize );
	bli_pthread_switch_off( &ind_l_state,    bli_ind_finalize );
	bli_pthread_switch_off( &gks_g_state,    bli_gks_finalize );

	return 0;
}

