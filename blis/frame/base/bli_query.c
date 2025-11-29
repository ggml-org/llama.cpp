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

bool bli_obj_equals( const obj_t* a, const obj_t* b )
{
#if 0
	bool  r_val = FALSE;
	num_t dt_a;
	num_t dt_b;
	num_t dt;

	// The function is not yet implemented for vectors and matrices.
	if ( !bli_obj_is_1x1( a ) ||
	     !bli_obj_is_1x1( b ) )
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

	dt_a = bli_obj_dt( a );
	dt_b = bli_obj_dt( b );

	// If B is BLIS_CONSTANT, then we need to test equality based on the
	// datatype of A--this works even if A is also BLIS_CONSTANT. If B
	// is a regular non-constant type, then we should use its datatype
	// to test equality.
	if ( dt_b == BLIS_CONSTANT ) dt = dt_a;
	else                         dt = dt_b;

	// Now test equality based on the chosen datatype.
	if ( dt == BLIS_CONSTANT )
	{
		dcomplex* ap_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, a );
		dcomplex* bp_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, b );

		// We only test equality for one datatype (double complex) since
		// we expect either all fields within the constant to be equal or
		// none to be equal. Therefore, we can just test one of them.
		r_val = bli_zeqa( ap_z, bp_z );
	}
	else
	{
		void* buf_a = bli_obj_buffer_for_1x1( dt, a );
		void* buf_b = bli_obj_buffer_for_1x1( dt, b );

		if      ( dt == BLIS_FLOAT )    r_val = bli_seqa( buf_a, buf_b );
		else if ( dt == BLIS_DOUBLE )   r_val = bli_deqa( buf_a, buf_b );
		else if ( dt == BLIS_SCOMPLEX ) r_val = bli_ceqa( buf_a, buf_b );
		else if ( dt == BLIS_DCOMPLEX ) r_val = bli_zeqa( buf_a, buf_b );
		else if ( dt == BLIS_INT )      r_val = bli_ieq( buf_a, buf_b );
	}

	return r_val;
#else
	bool r_val;

	if ( bli_obj_is_1x1( a ) && bli_obj_is_1x1( b ) )
		bli_eqsc( a, b, &r_val );
	else if ( bli_obj_is_vector( a ) && bli_obj_is_vector( b ) )
		bli_eqv( a, b, &r_val );
	else
		bli_eqm( a, b, &r_val );

	return r_val;
#endif
}

bool bli_obj_imag_equals( const obj_t* a, const obj_t* b )
{
#if 0
	bool  r_val = FALSE;
	num_t dt_a;
	num_t dt_b;

	dt_a = bli_obj_dt( a );
	dt_b = bli_obj_dt( b );

	// The function is not yet implemented for vectors and matrices.
	if ( !bli_obj_is_1x1( a ) ||
	     !bli_obj_is_1x1( b ) ||
	     bli_is_constant( dt_a ) ||
	     bli_is_complex( dt_b ) )
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

	// Handle the special (trivial) case where a is real, in which
	// case all we have to do is test whether b is zero.
	if ( bli_is_real( dt_a ) )
	{
		r_val = bli_obj_equals( &BLIS_ZERO, b );
	}
	else // if ( bli_is_complex( dt_a ) )
	{
		num_t dt_a_real = bli_dt_proj_to_real( dt_a );

		// Now we compare the imaginary part of a to b. Notice that since
		// we are using bli_obj_buffer_for_1x1() to acquire the buffer for
		// b, this works regardless of whether b is BLIS_CONSTANT.
		if ( dt_a == BLIS_SCOMPLEX )
		{
			scomplex* ap_c = bli_obj_buffer_at_off( a );
			float*    bp_c = bli_obj_buffer_for_1x1( dt_a_real, b );

			r_val = bli_seq( bli_cimag( *ap_c ), *bp_c );
		}
		else if ( dt_a == BLIS_DCOMPLEX )
		{
			dcomplex* ap_z = bli_obj_buffer_at_off( a );
			double*   bp_z = bli_obj_buffer_for_1x1( dt_a_real, b );

			r_val = bli_deq( bli_zimag( *ap_z ), *bp_z );
		}
	}
#endif
	bool r_val = FALSE;

	// The function is not yet implemented for vectors and matrices.
	if ( !bli_obj_is_1x1( a ) ||
	     !bli_obj_is_1x1( b ) ||
	     bli_obj_is_complex( b ) )
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

	double a_r, a_i;
	double b_r, b_i;

	// Get the real and imaginary parts of a and cast them to local doubles.
	bli_getsc( a, &a_r, &a_i );

	// Get the value of b and cast to a local double. (Note: the imaginary part
	// of b is ignored since we know b is real.)
	bli_getsc( b, &b_r, &b_i );

	// Compare the imaginary part of a to the real part of b.
	if ( a_i == b_r ) r_val = TRUE;

	return r_val;
}

bool bli_obj_imag_is_zero( const obj_t* a )
{
	bool r_val = TRUE;

	// The function is not yet implemented for vectors and matrices.
	if ( !bli_obj_is_1x1( a ) )
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

	if ( bli_obj_is_complex( a ) )
	{
		double a_r, a_i;

		// Get the real and imaginary parts and cast them to local doubles.
		bli_getsc( a, &a_r, &a_i );

		// Compare the imaginary part of a to double-precision zero.
		if ( !bli_teq0s( d, a_i ) ) r_val = FALSE;
	}

	return r_val;
}


