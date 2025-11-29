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

void bli_obj_create
     (
       num_t  dt,
       dim_t  m,
       dim_t  n,
       inc_t  rs,
       inc_t  cs,
       obj_t* obj
     )
{
	bli_init_once();

	bli_obj_create_without_buffer( dt, m, n, obj );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_obj_create(): " );
	#endif

	bli_obj_alloc_buffer( rs, cs, 1, obj );
}

void bli_obj_create_with_attached_buffer
     (
       num_t  dt,
       dim_t  m,
       dim_t  n,
       void*  p,
       inc_t  rs,
       inc_t  cs,
       obj_t* obj
     )
{
	bli_init_once();

	bli_obj_create_without_buffer( dt, m, n, obj );

	bli_obj_attach_buffer( p, rs, cs, 1, obj );
}

void bli_obj_create_without_buffer
     (
       num_t  dt,
       dim_t  m,
       dim_t  n,
       obj_t* obj
     )
{
	bli_init_once();

	if ( bli_error_checking_is_enabled() )
		bli_obj_create_without_buffer_check( dt, m, n, obj );

	// Query the size of one element of the object's pre-set datatype.
	siz_t elem_size = bli_dt_size( dt );

	// Set any default properties that are appropriate.
	bli_obj_set_defaults( obj );

	// Set the object root to itself, since obj is not presumed to be a view
	// into a larger matrix. This is typically the only time this field is
	// ever set; henceforth, subpartitions and aliases to this object will
	// get copies of this field, and thus always have access to its
	// "greatest-grand" parent (ie: the original parent, or "root", object).
	// However, there ARE a few places where it is convenient to reset the
	// root field explicitly via bli_obj_set_as_root(). (We do not list
	// those places here. Just grep for bli_obj_set_as_root within the
	// top-level 'frame' directory to see them.
	bli_obj_set_as_root( obj );

	// Set individual fields.
	bli_obj_set_buffer( NULL, obj );
	bli_obj_set_dt( dt, obj );
	bli_obj_set_elem_size( elem_size, obj );
	bli_obj_set_comp_prec( bli_dt_prec( dt ), obj );
	bli_obj_set_dims( m, n, obj );
	bli_obj_set_offs( 0, 0, obj );
	bli_obj_set_diag_offset( 0, obj );

	// Set the internal scalar to 1.0.
	bli_obj_set_scalar_dt( dt, obj );
	void* s = bli_obj_internal_scalar_buffer( obj );

	// Always writing the imaginary component is needed in mixed-domain
	// scenarios. Failing to do this can lead to reading uninitialized
	// memory just before calling the macrokernel (as the internal scalars
	// for A and B are merged).
	//if      ( bli_is_float( dt )    ) { bli_tset1s( s, *(( float*    )s) ); }
	//else if ( bli_is_double( dt )   ) { bli_tset1s( d, *(( double*   )s) ); }
	if      ( bli_is_float( dt )    ) { bli_tset1s( c, *(( scomplex* )s) ); }
	else if ( bli_is_double( dt )   ) { bli_tset1s( z, *(( dcomplex* )s) ); }
	else if ( bli_is_scomplex( dt ) ) { bli_tset1s( c, *(( scomplex* )s) ); }
	else if ( bli_is_dcomplex( dt ) ) { bli_tset1s( z, *(( dcomplex* )s) ); }
}

void bli_obj_alloc_buffer
     (
       inc_t  rs,
       inc_t  cs,
       inc_t  is,
       obj_t* obj
     )
{
	dim_t n_elem = 0;
	err_t r_val;

	bli_init_once();

	// Query the dimensions of the object we are allocating.
	dim_t m = bli_obj_length( obj );
	dim_t n = bli_obj_width( obj );

	// Query the size of one element.
	siz_t elem_size = bli_obj_elem_size( obj );

	// Adjust the strides, if needed, before doing anything else
	// (particularly, before doing any error checking).
	bli_adjust_strides( m, n, elem_size, &rs, &cs, &is );

	if ( bli_error_checking_is_enabled() )
		bli_obj_alloc_buffer_check( rs, cs, is, obj );

	// Determine how much object to allocate.
	if ( m == 0 || n == 0 )
	{
		// For empty objects, set n_elem to zero. Row and column strides
		// should remain unchanged (because alignment is not needed).
		n_elem = 0;
	}
	else
	{
		// The number of elements to allocate is given by the distance from
		// the element with the lowest address (usually {0, 0}) to the element
		// with the highest address (usually {m-1, n-1}), plus one for the
		// highest element itself.
		n_elem = (m-1) * bli_abs( rs ) + (n-1) * bli_abs( cs ) + 1;
	}

	// Handle the special case where imaginary stride is larger than
	// normal.
	if ( bli_obj_is_complex( obj ) )
	{
		// Notice that adding is/2 works regardless of whether the
		// imaginary stride is unit, something between unit and
		// 2*n_elem, or something bigger than 2*n_elem.
		n_elem = bli_abs( is ) / 2 + n_elem;
	}

	// Compute the size of the total buffer to be allocated, which includes
	// padding if the leading dimension was increased for alignment purposes.
	siz_t buffer_size = ( siz_t )n_elem * elem_size;

	// Allocate the buffer.
	void* p = bli_malloc_user( buffer_size, &r_val );

	// Set individual fields.
	bli_obj_set_buffer( p, obj );
	bli_obj_set_strides( rs, cs, obj );
	bli_obj_set_imag_stride( is, obj );
}

void bli_obj_attach_buffer
     (
       void*  p,
       inc_t  rs,
       inc_t  cs,
       inc_t  is,
       obj_t* obj
     )
{
	bli_init_once();

	// Interpret is = 0 as a request for the default, which is is = 1;
	if ( is == 0 ) is = 1;

	// Check that the strides and lengths are compatible. Note that the
	// user *must* specify valid row and column strides when attaching an
	// external buffer.
	if ( bli_error_checking_is_enabled() )
		bli_obj_attach_buffer_check( p, rs, cs, is, obj );

	// Update the object.
	bli_obj_set_buffer( p, obj );
	bli_obj_set_strides( rs, cs, obj );
	bli_obj_set_imag_stride( is, obj );
}

void bli_obj_create_1x1
     (
       num_t  dt,
       obj_t* obj
     )
{
	bli_obj_create_without_buffer( dt, 1, 1, obj );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_obj_create_1x1(): " );
	#endif

	bli_obj_alloc_buffer( 1, 1, 1, obj );
}

void bli_obj_create_1x1_with_attached_buffer
     (
       num_t  dt,
       void*  p,
       obj_t* obj
     )
{
	bli_obj_create_without_buffer( dt, 1, 1, obj );

	bli_obj_attach_buffer( p, 1, 1, 1, obj );
}

void bli_obj_create_conf_to
     (
       const obj_t* s,
             obj_t* d
     )
{
	const num_t dt = bli_obj_dt( s );
	const dim_t m  = bli_obj_length( s );
	const dim_t n  = bli_obj_width( s );
	const inc_t rs = bli_obj_row_stride( s );
	const inc_t cs = bli_obj_col_stride( s );

	bli_obj_create( dt, m, n, rs, cs, d );
}

void bli_obj_free
     (
       obj_t* obj
     )
{
	if ( bli_error_checking_is_enabled() )
		bli_obj_free_check( obj );

	// Don't dereference obj if it is NULL.
	if ( obj != NULL )
	{
		// Idiot safety: Don't try to free the buffer field if the object
		// is a detached scalar (ie: if the buffer pointer refers to the
		// address of the internal scalar buffer).
		if ( bli_obj_buffer( obj ) != bli_obj_internal_scalar_buffer( obj ) )
		{
			#ifdef BLIS_ENABLE_MEM_TRACING
			printf( "bli_obj_free(): " );
			#endif

			bli_free_user( bli_obj_buffer( obj ) );
		}
	}
}

#if 0
//void bli_obj_create_const
     (
       double value,
       obj_t* obj
     )
{
	gint_t*   temp_i;
	float*    temp_s;
	double*   temp_d;
	scomplex* temp_c;
	dcomplex* temp_z;

	if ( bli_error_checking_is_enabled() )
		bli_obj_create_const_check( value, obj );

	bli_obj_create( BLIS_CONSTANT, 1, 1, 1, 1, obj );

	//temp_s = bli_obj_buffer_for_const( BLIS_FLOAT,    obj );
	//temp_d = bli_obj_buffer_for_const( BLIS_DOUBLE,   obj );
	//temp_c = bli_obj_buffer_for_const( BLIS_SCOMPLEX, obj );
	//temp_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, obj );
	//temp_i = bli_obj_buffer_for_const( BLIS_INT,      obj );

	bli_tsets( d,s, value, 0.0, *temp_s );
	bli_tsets( d,d, value, 0.0, *temp_d );
	bli_tsets( d,c, value, 0.0, *temp_c );
	bli_tsets( d,z, value, 0.0, *temp_z );

	*temp_i = ( gint_t ) value;
}

//void bli_obj_create_const_copy_of
     (
       obj_t* a,
       obj_t* b
     )
{
	gint_t*   temp_i;
	float*    temp_s;
	double*   temp_d;
	scomplex* temp_c;
	dcomplex* temp_z;
	void*     buf_a;
	dcomplex  value;

	if ( bli_error_checking_is_enabled() )
		bli_obj_create_const_copy_of_check( a, b );

	bli_obj_create( BLIS_CONSTANT, 1, 1, 1, 1, b );

	//temp_s = bli_obj_buffer_for_const( BLIS_FLOAT,    b );
	//temp_d = bli_obj_buffer_for_const( BLIS_DOUBLE,   b );
	//temp_c = bli_obj_buffer_for_const( BLIS_SCOMPLEX, b );
	//temp_z = bli_obj_buffer_for_const( BLIS_DCOMPLEX, b );
	//temp_i = bli_obj_buffer_for_const( BLIS_INT,      b );

	buf_a = bli_obj_buffer_at_off( a );

	bli_tsets( z,z, 0.0, 0.0, value );

	if ( bli_obj_is_float( a ) )
	{
		bli_tcopys( s,z, *(( float*    )buf_a), value );
	}
	else if ( bli_obj_is_double( a ) )
	{
		bli_tcopys( d,z, *(( double*   )buf_a), value );
	}
	else if ( bli_obj_is_scomplex( a ) )
	{
		bli_tcopys( c,z, *(( scomplex* )buf_a), value );
	}
	else if ( bli_obj_is_dcomplex( a ) )
	{
		bli_tcopys( z,z, *(( dcomplex* )buf_a), value );
	}
	else
	{
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
	}

	bli_tcopys( z,s, value, *temp_s );
	bli_tcopys( z,d, value, *temp_d );
	bli_tcopys( z,c, value, *temp_c );
	bli_tcopys( z,z, value, *temp_z );

	*temp_i = ( gint_t ) bli_zreal( value );
}
#endif

void bli_adjust_strides
     (
       dim_t  m,
       dim_t  n,
       siz_t  elem_size,
       inc_t* rs,
       inc_t* cs,
       inc_t* is
     )
{
	// Here, we check the strides that were input from the user and modify
	// them if needed.

	// Handle the special "empty" case first. If either dimension is zero,
	// do nothing (this could represent a zero-length "slice" of another
	// matrix).
	if ( m == 0 || n == 0 ) return;

	// Interpret rs = cs = 0 as request for column storage and -1 as a request
	// for row storage.
	if ( *rs == 0 && *cs == 0 && ( *is == 0 || *is == 1 ) )
	{
		// First we handle the 1x1 scalar case explicitly.
		if ( m == 1 && n == 1 )
		{
			*rs = 1;
			*cs = 1;
		}
		// We use column-major storage, except when m == 1, in which case we
		// use what amounts to row-major storage because we don't want both
		// strides to be unit.
		else if ( m == 1 && n > 1 )
		{
			*rs = n;
			*cs = 1;
		}
		else
		{
			*rs = 1;
			*cs = m;
		}

		// Use default complex storage.
		*is = 1;

		// Align the strides depending on the tilt of the matrix. Note that
		// scalars are neither row nor column tilted. Also note that alignment
		// is only done for rs = cs = 0, and any user-supplied row and column
		// strides are preserved.
		if ( bli_is_col_tilted( m, n, *rs, *cs ) )
		{
			*cs = bli_align_dim_to_size( *cs, elem_size,
			                             BLIS_HEAP_STRIDE_ALIGN_SIZE );
		}
		else if ( bli_is_row_tilted( m, n, *rs, *cs ) )
		{
			*rs = bli_align_dim_to_size( *rs, elem_size,
		                                 BLIS_HEAP_STRIDE_ALIGN_SIZE );
		}
	}
	else if ( *rs == -1 && *cs == -1 && ( *is == 0 || *is == 1 ) )
	{
		// First we handle the 1x1 scalar case explicitly.
		if ( m == 1 && n == 1 )
		{
			*rs = 1;
			*cs = 1;
		}
		// We use row-major storage, except when n == 1, in which case we
		// use what amounts to column-major storage because we don't want both
		// strides to be unit.
		else if ( n == 1 && m > 1 )
		{
			*rs = 1;
			*cs = m;
		}
		else
		{
			*rs = n;
			*cs = 1;
		}

		// Use default complex storage.
		*is = 1;

		// Align the strides depending on the tilt of the matrix. Note that
		// scalars are neither row nor column tilted. Also note that alignment
		// is only done for rs = cs = -1, and any user-supplied row and column
		// strides are preserved.
		if ( bli_is_col_tilted( m, n, *rs, *cs ) )
		{
			*cs = bli_align_dim_to_size( *cs, elem_size,
			                             BLIS_HEAP_STRIDE_ALIGN_SIZE );
		}
		else if ( bli_is_row_tilted( m, n, *rs, *cs ) )
		{
			*rs = bli_align_dim_to_size( *rs, elem_size,
		                                 BLIS_HEAP_STRIDE_ALIGN_SIZE );
		}
	}
	else if ( *rs == 1 && *cs == 1 )
	{
		// If both strides are unit, this is probably a "lazy" request for a
		// single vector (but could also be a request for a 1xn matrix in
		// column-major order or an mx1 matrix in row-major order). In BLIS,
		// we have decided to "reserve" the case where rs = cs = 1 for
		// 1x1 scalars only.
		if ( m > 1 && n == 1 )
		{
			// Set the column stride to indicate that this is a column vector
			// stored in column-major order. This is done for legacy reasons,
			// because we at one time we had to satisify the error checking
			// in the underlying BLAS library, which expects the leading
			// dimension to be set to at least m, even if it will never be
			// used for indexing since it is a vector and thus only has one
			// column of data.
			*cs = m;
		}
		else if ( m == 1 && n > 1 )
		{
			// Set the row stride to indicate that this is a row vector stored
			// in row-major order.
			*rs = n;
		}

		// Nothing needs to be done for the 1x1 scalar case where m == n == 1.
	}
}

static siz_t dt_sizes[6] =
{
	sizeof( float ),
	sizeof( scomplex ),
	sizeof( double ),
	sizeof( dcomplex ),
	sizeof( gint_t ),
	sizeof( constdata_t )
};

siz_t bli_dt_size
     (
       num_t dt
     )
{
	if ( bli_error_checking_is_enabled() )
		bli_dt_size_check( dt );

	return dt_sizes[dt];
}

static char* dt_names[ BLIS_NUM_FP_TYPES+1 ] =
{
	"float",
	"scomplex",
	"double",
	"dcomplex",
	"int"
};

const char* bli_dt_string
     (
       num_t dt
     )
{
	if ( bli_error_checking_is_enabled() )
		bli_dt_string_check( dt );

	return dt_names[dt];
}

dim_t bli_align_dim_to_mult
     (
       dim_t dim,
       dim_t dim_mult,
       bool  round_up
     )
{
	// We return the dimension unmodified if the multiple is zero
	// (to avoid division by zero).
	if ( dim_mult == 0 ) return dim;

	if ( round_up )
		dim += dim_mult - 1;

	// Avoid rounding down to zero.
	return bli_max( dim_mult, ( dim / dim_mult ) * dim_mult );
}

dim_t bli_align_dim_to_size
     (
       dim_t dim,
       siz_t elem_size,
       siz_t align_size
     )
{
	dim = ( ( dim * ( dim_t )elem_size +
	                ( dim_t )align_size - 1
	        ) /
	        ( dim_t )align_size
	        ) *
	        ( dim_t )align_size /
	        ( dim_t )elem_size;

	return dim;
}

dim_t bli_align_ptr_to_size
     (
       const void*  p,
             size_t align_size
     )
{
	dim_t dim = ( ( ( uintptr_t )p + align_size - 1 ) /
	              align_size
	            ) * align_size;

	return dim;
}

#if 0
static num_t type_union[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES] =
{
            // s             c              d              z
	/* s */ { BLIS_FLOAT,    BLIS_SCOMPLEX, BLIS_DOUBLE,   BLIS_DCOMPLEX },
	/* c */ { BLIS_SCOMPLEX, BLIS_SCOMPLEX, BLIS_DCOMPLEX, BLIS_DCOMPLEX },
	/* d */ { BLIS_DOUBLE,   BLIS_DCOMPLEX, BLIS_DOUBLE,   BLIS_DCOMPLEX },
	/* z */ { BLIS_DCOMPLEX, BLIS_DCOMPLEX, BLIS_DCOMPLEX, BLIS_DCOMPLEX }
};

num_t bli_dt_union( num_t dt1, num_t dt2 )
{
	if ( bli_error_checking_is_enabled() )
		bli_dt_union_check( dt1, dt2 );

	return type_union[dt1][dt2];
}
#endif

void bli_obj_print
     (
       const char*  label,
       const obj_t* obj
     )
{
	bli_init_once();

	FILE* file = stdout;

	if ( bli_error_checking_is_enabled() )
		bli_obj_print_check( label, obj );

	fprintf( file, "\n" );
	fprintf( file, "%s\n", label );
	fprintf( file, "\n" );

	fprintf( file, " m x n           %lu x %lu\n", ( unsigned long )bli_obj_length( obj ),
	                                               ( unsigned long )bli_obj_width( obj ) );
	fprintf( file, "\n" );

	fprintf( file, " offm, offn      %lu, %lu\n", ( unsigned long )bli_obj_row_off( obj ),
	                                              ( unsigned long )bli_obj_col_off( obj ) );
	fprintf( file, " diagoff         %ld\n", ( signed long int )bli_obj_diag_offset( obj ) );
	fprintf( file, "\n" );

	fprintf( file, " buf             %p\n",  ( void* )bli_obj_buffer( obj ) );
	fprintf( file, " elem size       %lu\n", ( unsigned long )bli_obj_elem_size( obj ) );
	fprintf( file, " rs, cs          %ld, %ld\n", ( signed long int )bli_obj_row_stride( obj ),
	                                              ( signed long int )bli_obj_col_stride( obj ) );
	fprintf( file, " is              %ld\n", ( signed long int )bli_obj_imag_stride( obj ) );
	fprintf( file, " m_padded        %lu\n", ( unsigned long )bli_obj_padded_length( obj ) );
	fprintf( file, " n_padded        %lu\n", ( unsigned long )bli_obj_padded_width( obj ) );
	fprintf( file, " pd              %lu\n", ( unsigned long )bli_obj_panel_dim( obj ) );
	fprintf( file, " ps              %lu\n", ( unsigned long )bli_obj_panel_stride( obj ) );
	fprintf( file, "\n" );

	fprintf( file, " info            %lX\n", ( unsigned long )(*obj).info );
	fprintf( file, " - is complex    %lu\n", ( unsigned long )bli_obj_is_complex( obj ) );
	fprintf( file, " - is d. prec    %lu\n", ( unsigned long )bli_obj_is_double_prec( obj ) );
	fprintf( file, " - datatype      %lu\n", ( unsigned long )bli_obj_dt( obj ) );
	fprintf( file, " - comp prec     %lu\n", ( unsigned long )bli_obj_comp_prec( obj ) );
	fprintf( file, " - scalar dt     %lu\n", ( unsigned long )bli_obj_scalar_dt( obj ) );
	fprintf( file, " - has trans     %lu\n", ( unsigned long )bli_obj_has_trans( obj ) );
	fprintf( file, " - has conj      %lu\n", ( unsigned long )bli_obj_has_conj( obj ) );
	fprintf( file, " - unit diag?    %lu\n", ( unsigned long )bli_obj_has_unit_diag( obj ) );
	fprintf( file, " - struc type    %lu\n", ( unsigned long )bli_obj_struc( obj ) >> BLIS_STRUC_SHIFT );
	fprintf( file, " - uplo type     %lu\n", ( unsigned long )bli_obj_uplo( obj ) >> BLIS_UPLO_SHIFT );
	fprintf( file, "   - is upper    %lu\n", ( unsigned long )bli_obj_is_upper( obj ) );
	fprintf( file, "   - is lower    %lu\n", ( unsigned long )bli_obj_is_lower( obj ) );
	fprintf( file, "   - is dense    %lu\n", ( unsigned long )bli_obj_is_dense( obj ) );
	fprintf( file, " - pack schema   %lu\n", ( unsigned long )bli_obj_pack_schema( obj ) >> BLIS_PACK_SCHEMA_SHIFT );
	fprintf( file, " - packinv diag? %lu\n", ( unsigned long )bli_obj_has_inverted_diag( obj ) );
	fprintf( file, " - pack ordifup  %lu\n", ( unsigned long )bli_obj_is_pack_rev_if_upper( obj ) );
	fprintf( file, " - pack ordiflo  %lu\n", ( unsigned long )bli_obj_is_pack_rev_if_lower( obj ) );
	fprintf( file, " - packbuf type  %lu\n", ( unsigned long )bli_obj_pack_buffer_type( obj ) >> BLIS_PACK_BUFFER_SHIFT );
	fprintf( file, "\n" );
}

