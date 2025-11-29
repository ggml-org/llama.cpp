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

// Statically initialize structs containing representations of various
// constants for each datatype supported in BLIS.
static constdata_t bli_two_buffer   = bli_obj_init_constdata(  2.0 );
static constdata_t bli_one_buffer   = bli_obj_init_constdata(  1.0 );
static constdata_t bli_zero_buffer  = bli_obj_init_constdata(  0.0 );
static constdata_t bli_mone_buffer  = bli_obj_init_constdata( -1.0 );
static constdata_t bli_mtwo_buffer  = bli_obj_init_constdata( -2.0 );
static constdata_t bli_onei_buffer  = bli_obj_init_constdata_ri( 0.0,  1.0 );
static constdata_t bli_monei_buffer = bli_obj_init_constdata_ri( 0.0, -1.0 );
static constdata_t bli_nan_buffer   = bli_obj_init_constdata_ri( NAN,  NAN );

// Statically initialize global scalar constants, attaching the addresses
// of the corresponding structs above.
const obj_t BLIS_TWO         = bli_obj_init_const( &bli_two_buffer );
const obj_t BLIS_ONE         = bli_obj_init_const( &bli_one_buffer );
const obj_t BLIS_ZERO        = bli_obj_init_const( &bli_zero_buffer );
const obj_t BLIS_MINUS_ONE   = bli_obj_init_const( &bli_mone_buffer );
const obj_t BLIS_MINUS_TWO   = bli_obj_init_const( &bli_mtwo_buffer );
const obj_t BLIS_ONE_I       = bli_obj_init_const( &bli_onei_buffer );
const obj_t BLIS_MINUS_ONE_I = bli_obj_init_const( &bli_monei_buffer );
const obj_t BLIS_NAN         = bli_obj_init_const( &bli_nan_buffer );

#if 0
obj_t BLIS_TWO = {};
obj_t BLIS_ONE = {};
obj_t BLIS_ZERO = {};
obj_t BLIS_MINUS_ONE = {};
obj_t BLIS_MINUS_TWO = {};

void bli_const_init( void )
{
	bli_obj_create_const(  2.0, &BLIS_TWO );
	bli_obj_create_const(  1.0, &BLIS_ONE );
	bli_obj_create_const(  0.5, &BLIS_ONE_HALF );
	bli_obj_create_const(  0.0, &BLIS_ZERO );
	bli_obj_create_const( -0.5, &BLIS_MINUS_ONE_HALF );
	bli_obj_create_const( -1.0, &BLIS_MINUS_ONE );
	bli_obj_create_const( -2.0, &BLIS_MINUS_TWO );
}

void bli_const_finalize( void )
{
	bli_obj_free( &BLIS_TWO );
	bli_obj_free( &BLIS_ONE );
	bli_obj_free( &BLIS_ONE_HALF );
	bli_obj_free( &BLIS_ZERO );
	bli_obj_free( &BLIS_MINUS_ONE_HALF );
	bli_obj_free( &BLIS_MINUS_ONE );
	bli_obj_free( &BLIS_MINUS_TWO );
}
#endif

