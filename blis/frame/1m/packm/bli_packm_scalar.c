/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Southern Methodist University

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

void* bli_packm_scalar( obj_t* kappa, obj_t* p )
{
	num_t dt_p = bli_obj_dt( p );

	// The value for kappa we use will depends on whether the scalar
	// attached to A has a nonzero imaginary component. If it does,
	// and the matrix to pack is complex, then we apply the scalar now
	// because we may not have a chance later due to using real-domain
	// microkernels.
	if ( bli_obj_scalar_has_nonzero_imag( p ) &&
	     ( bli_obj_is_complex( p ) ||
	       bli_obj_pack_schema( p ) == BLIS_PACKED_PANELS_RO ) )
	{
		//printf( "applying non-zero imag kappa\n_p" );

		// Detach the scalar.
		bli_obj_scalar_detach( p, kappa );

		// Reset the attached scalar (to 1.0).
		bli_obj_scalar_reset( p );

		return bli_obj_buffer_for_1x1( dt_p, kappa );
	}
	else
	{
		// If the internal scalar of A has only a real component, then
		// we will apply it later (in the micro-kernel), and so we will
		// use BLIS_ONE to indicate no scaling during packing.
		return bli_obj_buffer_for_1x1( dt_p, &BLIS_ONE );
	}
}

