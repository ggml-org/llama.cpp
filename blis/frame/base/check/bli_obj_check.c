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

void bli_obj_create_check(       num_t  dt,
                                 dim_t  m,
                                 dim_t  n,
                                 inc_t  rs,
                                 inc_t  cs,
                           const obj_t* obj )
{
	err_t e_val;

	e_val = bli_check_valid_datatype( dt );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_strides( m, n, rs, cs, 1 );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( obj );
	bli_check_error_code( e_val );
}

void bli_obj_create_without_buffer_check(       num_t  dt,
                                                dim_t  m,
                                                dim_t  n,
                                          const obj_t* obj )
{
	err_t e_val;

	e_val = bli_check_valid_datatype( dt );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( obj );
	bli_check_error_code( e_val );
}

void bli_obj_alloc_buffer_check(       inc_t  rs,
                                       inc_t  cs,
                                       inc_t  is,
                                 const obj_t* obj )
{
	err_t e_val;

	e_val = bli_check_matrix_strides( bli_obj_length( obj ),
	                                  bli_obj_width( obj ),
	                                  rs, cs, is );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( obj );
	bli_check_error_code( e_val );
}

void bli_obj_attach_buffer_check( const void*  p,
                                        inc_t  rs,
                                        inc_t  cs,
                                        inc_t  is,
                                  const obj_t* obj )
{
	err_t e_val;

	// NOTE: We allow the caller to attach NULL to an object because
	// the buffer contains NULL after _create_wihout_buffer() anyway.
	// Thus, we're not opening a window for undefined behavior because
	// that window is already open. Instead of checking for NULL here,
	// we check the object buffers for all objects in all of the
	// computational operations' _check()/_int_check() functions.
	//e_val = bli_check_null_pointer( p );
	//bli_check_error_code( e_val );

	e_val = bli_check_matrix_strides( bli_obj_length( obj ),
	                                  bli_obj_width( obj ),
	                                  rs, cs, is );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( obj );
	bli_check_error_code( e_val );
}

void bli_obj_create_scalar_check( num_t dt, const obj_t* obj )
{
	err_t e_val;

	e_val = bli_check_valid_datatype( dt );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( obj );
	bli_check_error_code( e_val );
}

void bli_obj_free_check( const obj_t* obj )
{
	//err_t e_val;

	// We don't bother checking for null-ness since bli_obj_free()
	// handles null pointers safely.
	//e_val = bli_check_null_pointer( obj );
	//bli_check_error_code( e_val );
}

void bli_obj_create_const_check( double value, const obj_t* obj )
{
	err_t e_val;

	e_val = bli_check_null_pointer( obj );
	bli_check_error_code( e_val );
}

#if 0
void bli_obj_create_const_copy_of_check( obj_t* a, obj_t* b )
{
	err_t e_val;

	e_val = bli_check_null_pointer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( b );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );
}
#endif

void bli_dt_size_check( num_t dt )
{
	err_t e_val;

	e_val = bli_check_valid_datatype( dt );
	bli_check_error_code( e_val );
}

void bli_dt_string_check( num_t dt )
{
	err_t e_val;

	e_val = bli_check_nonconstant_datatype( dt );
	bli_check_error_code( e_val );
}

void bli_dt_union_check( num_t dt1, num_t dt2 )
{
	err_t e_val;

	e_val = bli_check_floating_datatype( dt1 );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_datatype( dt2 );
	bli_check_error_code( e_val );
}

void bli_obj_print_check( const char* label, const obj_t* obj )
{
	err_t e_val;

	e_val = bli_check_null_pointer( label );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( obj );
	bli_check_error_code( e_val );
}

