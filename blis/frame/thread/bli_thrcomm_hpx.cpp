/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 Tactical Computing Laboratories, LLC

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

#ifdef BLIS_ENABLE_HPX

#include <hpx/synchronization/barrier.hpp>
extern "C" {

// Define the pthread_barrier_t implementations of the init, cleanup, and
// barrier functions.

void hpx_barrier_init( hpx_barrier_t* barrier, dim_t n_threads )
{
	if ( barrier == nullptr ) return;
	barrier->handle = new hpx::barrier<>( n_threads );
}

void hpx_barrier_destroy( hpx_barrier_t* barrier )
{
	if ( barrier == nullptr ) return;

	auto* barrier_ = reinterpret_cast<hpx::barrier<>*>( barrier->handle );
	barrier->handle = nullptr;

	delete barrier_; 
}

void hpx_barrier_arrive_and_wait( hpx_barrier_t* barrier )
{
	if ( barrier == nullptr ) return;
	auto* barrier_ = reinterpret_cast<hpx::barrier<>*>( barrier->handle );

	if ( barrier_ == nullptr ) return;
	barrier_->arrive_and_wait();
}

void bli_thrcomm_init_hpx( dim_t n_threads, thrcomm_t* comm )
{
	if ( comm == nullptr ) return;

	comm->sent_object             = nullptr;
	comm->n_threads               = n_threads;
	comm->ti                      = BLIS_HPX;
	// comm->barrier_sense           = 0;
	// comm->barrier_threads_arrived = 0;

	hpx_barrier_init( &comm->barrier, n_threads );
}

void bli_thrcomm_cleanup_hpx( thrcomm_t* comm )
{
	if ( comm == nullptr ) return;
	hpx_barrier_destroy( &comm->barrier );
}

void bli_thrcomm_barrier_hpx( dim_t t_id, thrcomm_t* comm )
{
	hpx_barrier_arrive_and_wait( &comm->barrier );
}

}

#endif

