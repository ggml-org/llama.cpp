/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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

void bli_pack_get_pack_a( bool* pack_a )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	*pack_a = bli_rntm_pack_a( bli_global_rntm() );
}

// -----------------------------------------------------------------------------

void bli_pack_get_pack_b( bool* pack_b )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	*pack_b = bli_rntm_pack_b( bli_global_rntm() );
}

// ----------------------------------------------------------------------------

void bli_pack_set_pack_a( bool pack_a )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	// If TLS is disabled, we need to use a mutex to protect the global rntm_t
	// since it will be shared with all application threads.
	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_lock( bli_global_rntm_mutex() );
	#endif

	bli_rntm_set_pack_a( pack_a, bli_global_rntm() );

	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_unlock( bli_global_rntm_mutex() );
	#endif
}

// ----------------------------------------------------------------------------

void bli_pack_set_pack_b( bool pack_b )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	// If TLS is disabled, we need to use a mutex to protect the global rntm_t
	// since it will be shared with all application threads.
	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_lock( bli_global_rntm_mutex() );
	#endif

	bli_rntm_set_pack_b( pack_b, bli_global_rntm() );

	#ifdef BLIS_DISABLE_TLS
	bli_pthread_mutex_unlock( bli_global_rntm_mutex() );
	#endif
}

