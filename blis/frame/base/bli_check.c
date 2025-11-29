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

// -- General stuff ------------------------------------------------------------

err_t bli_check_error_code_helper( gint_t code, const char* file, guint_t line )
{
	if ( code == BLIS_SUCCESS ) return code;

	if ( BLIS_ERROR_CODE_MAX < code && code < BLIS_ERROR_CODE_MIN )
	{
		bli_print_msg( bli_error_string_for_code( code ),
		               file, line );
		bli_abort();
	}
	else
	{
		bli_print_msg( bli_error_string_for_code( BLIS_UNDEFINED_ERROR_CODE ),
		               file, line );
		bli_abort();
	}

	return code;
}

err_t bli_check_valid_error_level( errlev_t level )
{
	err_t e_val = BLIS_SUCCESS;

	if ( level != BLIS_NO_ERROR_CHECKING &&
	     level != BLIS_FULL_ERROR_CHECKING )
		e_val = BLIS_INVALID_ERROR_CHECKING_LEVEL;

	return e_val;
}

err_t bli_check_null_pointer( const void* ptr )
{
	err_t e_val = BLIS_SUCCESS;

	if ( ptr == NULL )
		e_val = BLIS_NULL_POINTER;

	return e_val;
}

// -- Parameter-related checks -------------------------------------------------

err_t bli_check_valid_side( side_t side )
{
	err_t e_val = BLIS_SUCCESS;

	if ( side != BLIS_LEFT &&
	     side != BLIS_RIGHT /*&&
	     side != BLIS_TOP &&
	     side != BLIS_BOTTOM*/ )
		e_val = BLIS_INVALID_SIDE;

	return e_val;
}

err_t bli_check_valid_uplo( uplo_t uplo )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_is_lower( uplo ) &&
	     !bli_is_upper( uplo ) )
		e_val = BLIS_INVALID_UPLO;

	return e_val;
}

err_t bli_check_valid_trans( trans_t trans )
{
	err_t e_val = BLIS_SUCCESS;

	if ( trans != BLIS_NO_TRANSPOSE &&
	     trans != BLIS_TRANSPOSE &&
	     trans != BLIS_CONJ_NO_TRANSPOSE &&
	     trans != BLIS_CONJ_TRANSPOSE )
		e_val = BLIS_INVALID_TRANS;

	return e_val;
}

err_t bli_check_valid_diag( diag_t diag )
{
	err_t e_val = BLIS_SUCCESS;

	if ( diag != BLIS_NONUNIT_DIAG &&
	     diag != BLIS_UNIT_DIAG )
		e_val = BLIS_INVALID_DIAG;

	return e_val;
}

err_t bli_check_nonunit_diag( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_obj_has_nonunit_diag( a ) )
		e_val = BLIS_EXPECTED_NONUNIT_DIAG;

	return e_val;
}

// -- Datatype-related checks --------------------------------------------------

err_t bli_check_valid_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt != BLIS_FLOAT &&
	     dt != BLIS_DOUBLE &&
	     dt != BLIS_SCOMPLEX &&
	     dt != BLIS_DCOMPLEX &&
	     dt != BLIS_INT &&
	     dt != BLIS_CONSTANT )
		e_val = BLIS_INVALID_DATATYPE;

	return e_val;
}

err_t bli_check_object_valid_datatype( const obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bli_obj_dt( a );
	e_val = bli_check_valid_datatype( dt );

	return e_val;
}

err_t bli_check_noninteger_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt == BLIS_INT )
		e_val = BLIS_EXPECTED_NONINTEGER_DATATYPE;

	return e_val;
}

err_t bli_check_noninteger_object( const obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bli_obj_dt( a );
	e_val = bli_check_noninteger_datatype( dt );

	return e_val;
}

err_t bli_check_nonconstant_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt == BLIS_CONSTANT )
		e_val = BLIS_EXPECTED_NONCONSTANT_DATATYPE;

	return e_val;
}

err_t bli_check_nonconstant_object( const obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bli_obj_dt( a );
	e_val = bli_check_nonconstant_datatype( dt );

	return e_val;
}

err_t bli_check_floating_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt != BLIS_FLOAT &&
	     dt != BLIS_DOUBLE &&
	     dt != BLIS_SCOMPLEX &&
	     dt != BLIS_DCOMPLEX )
		e_val = BLIS_EXPECTED_FLOATING_POINT_DATATYPE;

	return e_val;
}

err_t bli_check_floating_object( const obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bli_obj_dt( a );
	e_val = bli_check_floating_datatype( dt );

	return e_val;
}

err_t bli_check_real_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt != BLIS_FLOAT &&
	     dt != BLIS_DOUBLE )
		e_val = BLIS_EXPECTED_REAL_DATATYPE;

	return e_val;
}

err_t bli_check_real_object( const obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bli_obj_dt( a );
	e_val = bli_check_real_datatype( dt );

	return e_val;
}

err_t bli_check_integer_datatype( num_t dt )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt != BLIS_INT )
		e_val = BLIS_EXPECTED_INTEGER_DATATYPE;

	return e_val;
}

err_t bli_check_integer_object( const obj_t* a )
{
	err_t e_val;
	num_t dt;

	dt = bli_obj_dt( a );
	e_val = bli_check_integer_datatype( dt );

	return e_val;
}

err_t bli_check_consistent_datatypes( num_t dt_a, num_t dt_b )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt_a != BLIS_CONSTANT &&
	     dt_b != BLIS_CONSTANT )
		if ( dt_a != dt_b )
			e_val = BLIS_INCONSISTENT_DATATYPES;

	return e_val;
}

err_t bli_check_consistent_object_datatypes( const obj_t* a, const obj_t* b )
{
	err_t e_val;
	num_t dt_a;
	num_t dt_b;

	dt_a = bli_obj_dt( a );
	dt_b = bli_obj_dt( b );

	e_val = bli_check_consistent_datatypes( dt_a, dt_b );

	return e_val;
}

err_t bli_check_datatype_real_proj_of( num_t dt_c, num_t dt_r )
{
	err_t e_val = BLIS_SUCCESS;

	if ( ( dt_c == BLIS_CONSTANT && bli_is_complex( dt_r ) ) ||
	     ( dt_c == BLIS_FLOAT    && dt_r != BLIS_FLOAT     ) ||
	     ( dt_c == BLIS_DOUBLE   && dt_r != BLIS_DOUBLE    ) ||
	     ( dt_c == BLIS_SCOMPLEX && dt_r != BLIS_FLOAT     ) ||
	     ( dt_c == BLIS_DCOMPLEX && dt_r != BLIS_DOUBLE    ) )
		e_val = BLIS_EXPECTED_REAL_PROJ_OF;

	return e_val;
}

err_t bli_check_object_real_proj_of( const obj_t* c, const obj_t* r )
{
	err_t e_val;
	num_t dt_c;
	num_t dt_r;

	dt_c = bli_obj_dt( c );
	dt_r = bli_obj_dt( r );

	e_val = bli_check_datatype_real_proj_of( dt_c, dt_r );

	return e_val;
}

err_t bli_check_real_valued_object( const obj_t* a )
{
	err_t  e_val = BLIS_SUCCESS;
	double a_real;
	double a_imag;

	bli_getsc( a, &a_real, &a_imag );

	if ( a_imag != 0.0 )
		e_val = BLIS_EXPECTED_REAL_VALUED_OBJECT;

	return e_val;
}

err_t bli_check_consistent_precisions( num_t dt_a, num_t dt_b )
{
	err_t e_val = BLIS_SUCCESS;

	if ( dt_a == BLIS_FLOAT )
	{
		if ( dt_b != BLIS_FLOAT &&
		     dt_b != BLIS_SCOMPLEX )
			e_val = BLIS_INCONSISTENT_PRECISIONS;
	}
	else if ( dt_a == BLIS_DOUBLE )
	{
		if ( dt_b != BLIS_DOUBLE &&
		     dt_b != BLIS_DCOMPLEX )
			e_val = BLIS_INCONSISTENT_PRECISIONS;
	}

	return e_val;
}

err_t bli_check_consistent_object_precisions( const obj_t* a, const obj_t* b )
{
	err_t e_val;
	num_t dt_a;
	num_t dt_b;

	dt_a = bli_obj_dt( a );
	dt_b = bli_obj_dt( b );

	e_val = bli_check_consistent_precisions( dt_a, dt_b );

	return e_val;
}

// -- Dimension-related checks -------------------------------------------------

err_t bli_check_conformal_dims( const obj_t* a, const obj_t* b )
{
	err_t e_val = BLIS_SUCCESS;
	dim_t m_a, n_a;
	dim_t m_b, n_b;

	m_a = bli_obj_length_after_trans( a );
	n_a = bli_obj_width_after_trans( a );
	m_b = bli_obj_length_after_trans( b );
	n_b = bli_obj_width_after_trans( b );

	if ( m_a != m_b || n_a != n_b )
		e_val = BLIS_NONCONFORMAL_DIMENSIONS;

	return e_val;
}

err_t bli_check_level3_dims( const obj_t* a, const obj_t* b, const obj_t* c )
{
	err_t e_val = BLIS_SUCCESS;
	dim_t m_c, n_c;
	dim_t m_a, k_a;
	dim_t k_b, n_b;

	m_c = bli_obj_length_after_trans( c );
	n_c = bli_obj_width_after_trans( c );

	m_a = bli_obj_length_after_trans( a );
	k_a = bli_obj_width_after_trans( a );

	k_b = bli_obj_length_after_trans( b );
	n_b = bli_obj_width_after_trans( b );

	if ( m_c != m_a ||
	     n_c != n_b ||
	     k_a != k_b )
		e_val = BLIS_NONCONFORMAL_DIMENSIONS;

	return e_val;
}

err_t bli_check_scalar_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_length( a ) < 0 ||
		 bli_obj_width( a )  < 0 )
		return BLIS_NEGATIVE_DIMENSION;

	if ( bli_obj_length( a ) != 1 ||
	     bli_obj_width( a )  != 1 )
		return BLIS_EXPECTED_SCALAR_OBJECT;

	return e_val;
}

err_t bli_check_vector_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_length( a ) < 0 ||
		 bli_obj_width( a )  < 0 )
		return BLIS_NEGATIVE_DIMENSION;

	if ( !bli_obj_is_vector( a ) )
		return BLIS_EXPECTED_VECTOR_OBJECT;

	return e_val;
}

err_t bli_check_matrix_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_length( a ) < 0 ||
		 bli_obj_width( a )  < 0 )
		e_val = BLIS_NEGATIVE_DIMENSION;

	return e_val;
}

err_t bli_check_equal_vector_lengths( const obj_t* x, const obj_t* y )
{
	err_t e_val = BLIS_SUCCESS;
	dim_t dim_x;
	dim_t dim_y;

	dim_x = bli_obj_vector_dim( x );
	dim_y = bli_obj_vector_dim( y );

	if ( dim_x != dim_y )
		e_val = BLIS_UNEQUAL_VECTOR_LENGTHS;

	return e_val;
}

err_t bli_check_square_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_length( a ) != bli_obj_width( a ) )
		e_val = BLIS_EXPECTED_SQUARE_OBJECT;

	return e_val;
}

err_t bli_check_object_length_equals( const obj_t* a, dim_t m )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_length( a ) != m )
		e_val = BLIS_UNEXPECTED_OBJECT_LENGTH;

	return e_val;
}

err_t bli_check_object_width_equals( const obj_t* a, dim_t n )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_width( a ) != n )
		e_val = BLIS_UNEXPECTED_OBJECT_WIDTH;

	return e_val;
}

err_t bli_check_vector_dim_equals( const obj_t* a, dim_t n )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_vector_dim( a ) != n )
		e_val = BLIS_UNEXPECTED_VECTOR_DIM;

	return e_val;
}

err_t bli_check_object_diag_offset_equals( const obj_t* a, doff_t offset )
{
	err_t e_val = BLIS_SUCCESS;

	if ( offset != bli_obj_diag_offset( a ) )
		e_val = BLIS_UNEXPECTED_DIAG_OFFSET;

	return e_val;
}

// -- Stride-related checks ----------------------------------------------------

err_t bli_check_matrix_strides( dim_t m, dim_t n, inc_t rs, inc_t cs, inc_t is )
{
	err_t e_val = BLIS_SUCCESS;

	// Note: A lot of thought went into designing these checks. Do NOT change
	// them unless you absolutely know what you are doing! Particularly, do
	// not try to merge the general and row-/column-major sections. It might
	// be possible, but it would be a lot less readable.

	// Prohibit negative dimensions.
	if ( m < 0 || n < 0 )
		return BLIS_NEGATIVE_DIMENSION;

	// Overwrite rs and cs with the absolute value of each. We can do this
	// since the checks below are not dependent on the sign of the strides.
	rs = bli_abs( rs );
	cs = bli_abs( cs );
	is = bli_abs( is );

	// The default case (whereby we interpret rs == cs == 0 as a request for
	// column-major order) is handled prior to calling this function, so the
	// only time we should see zero strides here is if the matrix is empty.
	if ( m == 0 || n == 0 ) return e_val;

	// Disallow row, column, or imaginary strides of zero.
	if ( ( rs == 0 || cs == 0 || is == 0 ) )
		return BLIS_INVALID_DIM_STRIDE_COMBINATION;

	// Check stride consistency in cases of general stride.
	if ( rs != 1 && cs != 1 )
	{
		// We apply different tests depending on which way the strides
		// "tilt".
		if ( rs == cs )
		{
			// If rs == cs, then we must be dealing with an m-by-1 or a
			// 1-by-n matrix and thus at least one of the dimensions, m
			// or n, must be unit (even if the other is zero).
			if ( m != 1 && n != 1 )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
		else if ( rs < cs )
		{
			// For column-major tilt, cs must be equal or larger than m * rs.
			if ( m * rs > cs )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
		else if ( cs < rs )
		{
			// For row-major tilt, rs must be equal or larger than n * cs.
			if ( n * cs > rs )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
	}
	else // check stride consistency of row-/column-storage cases.
	{
		if ( rs == 1 && cs == 1 )
		{
			// If rs == cs == 1, then we must be dealing with an m-by-1, a
			// 1-by-n, or a 1-by-1 matrix and thus at least one of the
			// dimensions, m or n, must be unit (even if the other is zero).
			if ( m != 1 && n != 1 )
				return BLIS_INVALID_DIM_STRIDE_COMBINATION;
		}
		else if ( rs == 1 )
		{
			// For column-major storage, don't allow the column stride to be
			// less than the m dimension.
			if ( cs < m )
				return BLIS_INVALID_COL_STRIDE;
		}
		else if ( cs == 1 )
		{
			// For row-major storage, don't allow the row stride to be less
			// than the n dimension.
			if ( rs < n )
				return BLIS_INVALID_ROW_STRIDE;
		}
	}

	return e_val;
}

// -- Structure-related checks -------------------------------------------------

err_t bli_check_general_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_obj_is_general( a ) )
		e_val = BLIS_EXPECTED_GENERAL_OBJECT;

	return e_val;
}

err_t bli_check_hermitian_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_obj_is_hermitian( a ) )
		e_val = BLIS_EXPECTED_HERMITIAN_OBJECT;

	return e_val;
}

err_t bli_check_symmetric_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_obj_is_symmetric( a ) )
		e_val = BLIS_EXPECTED_SYMMETRIC_OBJECT;

	return e_val;
}

err_t bli_check_triangular_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_obj_is_triangular( a ) )
		e_val = BLIS_EXPECTED_TRIANGULAR_OBJECT;

	return e_val;
}

err_t bli_check_object_struc( const obj_t* a, struc_t struc )
{
	err_t e_val = BLIS_SUCCESS;

	if      ( bli_is_general( struc ) )    e_val = bli_check_general_object( a );
	else if ( bli_is_hermitian( struc ) )  e_val = bli_check_hermitian_object( a );
	else if ( bli_is_symmetric( struc ) )  e_val = bli_check_symmetric_object( a );
	else if ( bli_is_triangular( struc ) ) e_val = bli_check_triangular_object( a );

	return e_val;
}

// -- Storage-related checks ---------------------------------------------------

err_t bli_check_upper_or_lower_object( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_obj_is_lower( a ) &&
	     !bli_obj_is_upper( a ) )
		e_val = BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT;

	return e_val;
}

// -- Partitioning-related checks ----------------------------------------------

err_t bli_check_valid_3x1_subpart( subpart_t part )
{
	err_t e_val = BLIS_SUCCESS;

	if ( part != BLIS_SUBPART0  &&
	     part != BLIS_SUBPART1AND0 &&
	     part != BLIS_SUBPART1  &&
	     part != BLIS_SUBPART1AND2 &&
	     part != BLIS_SUBPART2 &&
	     part != BLIS_SUBPART1A &&
	     part != BLIS_SUBPART1B )
		e_val = BLIS_INVALID_3x1_SUBPART;

	return e_val;
}

err_t bli_check_valid_1x3_subpart( subpart_t part )
{
	err_t e_val = BLIS_SUCCESS;

	if ( part != BLIS_SUBPART0  &&
	     part != BLIS_SUBPART1AND0 &&
	     part != BLIS_SUBPART1  &&
	     part != BLIS_SUBPART1AND2 &&
	     part != BLIS_SUBPART2 &&
	     part != BLIS_SUBPART1A &&
	     part != BLIS_SUBPART1B )
		e_val = BLIS_INVALID_1x3_SUBPART;

	return e_val;
}

err_t bli_check_valid_3x3_subpart( subpart_t part )
{
	err_t e_val = BLIS_SUCCESS;

	if ( part != BLIS_SUBPART00 &&
	     part != BLIS_SUBPART10 &&
	     part != BLIS_SUBPART20 &&
	     part != BLIS_SUBPART01 &&
	     part != BLIS_SUBPART11 &&
	     part != BLIS_SUBPART21 &&
	     part != BLIS_SUBPART02 &&
	     part != BLIS_SUBPART12 &&
	     part != BLIS_SUBPART22 )
		e_val = BLIS_INVALID_3x3_SUBPART;

	return e_val;
}

// -- Control tree-related checks ----------------------------------------------

err_t bli_check_valid_cntl( const void* cntl )
{
	err_t e_val = BLIS_SUCCESS;

	if ( cntl == NULL )
		e_val = BLIS_UNEXPECTED_NULL_CONTROL_TREE;

	return e_val;
}

// -- Packing-related checks ---------------------------------------------------

err_t bli_check_packm_schema_on_unpack( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_pack_schema( a ) != BLIS_PACKED_MATRIX &&
	     bli_obj_pack_schema( a ) != BLIS_PACKED_PANELS )
		e_val = BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK;

	return e_val;
}

err_t bli_check_packv_schema_on_unpack( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_obj_pack_schema( a ) != BLIS_PACKED_VECTOR )
		e_val = BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK;

	return e_val;
}

// -- Buffer-related checks ----------------------------------------------------

err_t bli_check_object_buffer( const obj_t* a )
{
	err_t e_val = BLIS_SUCCESS;

	// We are only concerned with NULL buffers in objects where BOTH
	// dimensions are non-zero.
	if ( bli_obj_buffer( a ) == NULL )
	if ( bli_obj_length( a ) > 0 && bli_obj_width( a ) > 0 )
		e_val = BLIS_EXPECTED_NONNULL_OBJECT_BUFFER;

	return e_val;
}

// -- Memory checks ------------------------------------------------------------

err_t bli_check_valid_malloc_buf( const void* ptr )
{
	err_t e_val = BLIS_SUCCESS;

	if ( ptr == NULL )
		e_val = BLIS_MALLOC_RETURNED_NULL;

	return e_val;
}


// -- Internal memory pool checks ----------------------------------------------

err_t bli_check_valid_packbuf( packbuf_t buf_type )
{
	err_t e_val = BLIS_SUCCESS;

	if ( buf_type != BLIS_BUFFER_FOR_A_BLOCK &&
	     buf_type != BLIS_BUFFER_FOR_B_PANEL &&
	     buf_type != BLIS_BUFFER_FOR_C_PANEL &&
	     buf_type != BLIS_BUFFER_FOR_GEN_USE )
		e_val = BLIS_INVALID_PACKBUF;

	return e_val;
}

err_t bli_check_if_exhausted_pool( const pool_t* pool )
{
	err_t e_val = BLIS_SUCCESS;

	if ( bli_pool_is_exhausted( pool ) )
		e_val = BLIS_EXHAUSTED_CONTIG_MEMORY_POOL;

	return e_val;
}

err_t bli_check_sufficient_stack_buf_size( const cntx_t* cntx )
{
	err_t e_val = BLIS_SUCCESS;
	num_t dt;

	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		dim_t mr      = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
		dim_t nr      = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
		// Always use the size of the largest data type to account for
		// conversions during mixed-domain/mixed-precision computation.
		siz_t dt_size = bli_dt_size( BLIS_DCOMPLEX );

		// NOTE: For induced methods, we use the size of the complex datatypes
		// (rather than the size of the native micro-kernels' datatype) because
		// the macro-kernel needs this larger micro-tile footprint, even if the
		// virtual micro-kernel implementation will only ever be writing to half
		// of it (real or imaginary part) at a time.

		if ( mr * nr * dt_size > BLIS_STACK_BUF_MAX_SIZE )
			e_val = BLIS_INSUFFICIENT_STACK_BUF_SIZE;
	}

	return e_val;
}

err_t bli_check_alignment_is_power_of_two( size_t align_size )
{
	err_t e_val = BLIS_SUCCESS;

	// This function returns an error code if align_size is zero or not
	// a power of two.

	if ( align_size == 0 )
		e_val = BLIS_ALIGNMENT_NOT_POWER_OF_TWO;
	else if ( ( align_size & ( align_size - 1 ) ) )
		e_val = BLIS_ALIGNMENT_NOT_POWER_OF_TWO;

	return e_val;
}

err_t bli_check_alignment_is_mult_of_ptr_size( size_t align_size )
{
	err_t e_val = BLIS_SUCCESS;

	// This function returns an error code if align_size is not a whole
	// multiple of the size of a pointer.

	if ( align_size % sizeof( void* ) != 0 )
		e_val = BLIS_ALIGNMENT_NOT_MULT_OF_PTR_SIZE;

	return e_val;
}

// -- Object-related errors ----------------------------------------------------

err_t bli_check_object_alias_of( const obj_t* a, const obj_t* b )
{
	err_t e_val = BLIS_SUCCESS;

	if ( !bli_obj_is_alias_of( a, b ) )
		e_val = BLIS_EXPECTED_OBJECT_ALIAS;

	return e_val;
}

// -- Architecture-related errors ----------------------------------------------

err_t bli_check_valid_arch_id( arch_t id )
{
	err_t e_val = BLIS_SUCCESS;

	if ( ( gint_t )id < 0 || BLIS_NUM_ARCHS <= ( gint_t )id )
		e_val = BLIS_INVALID_ARCH_ID;

	return e_val;
}

err_t bli_check_initialized_gks_cntx( const cntx_t* cntx )
{
	err_t e_val = BLIS_SUCCESS;

	if ( cntx == NULL )
		e_val = BLIS_UNINITIALIZED_GKS_CNTX;

	return e_val;
}

// -- Architecture-related errors ----------------------------------------------

err_t bli_check_valid_mc_mod_mult( const blksz_t* mc, const blksz_t* mr )
{
	num_t dt;

	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		dim_t mc_def_dt = bli_blksz_get_def( dt, mc );
		dim_t mc_max_dt = bli_blksz_get_max( dt, mc );
		dim_t mr_dt  = bli_blksz_get_def( dt, mr );

		if      ( mc_def_dt % mr_dt != 0 ) return BLIS_MC_DEF_NONMULTIPLE_OF_MR;
		else if ( mc_max_dt % mr_dt != 0 ) return BLIS_MC_MAX_NONMULTIPLE_OF_MR;
	}

	return BLIS_SUCCESS;
}

err_t bli_check_valid_nc_mod_mult( const blksz_t* nc, const blksz_t* nr )
{
	num_t dt;

	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		dim_t nc_def_dt = bli_blksz_get_def( dt, nc );
		dim_t nc_max_dt = bli_blksz_get_max( dt, nc );
		dim_t nr_dt     = bli_blksz_get_def( dt, nr );

		if      ( nc_def_dt % nr_dt != 0 ) return BLIS_NC_DEF_NONMULTIPLE_OF_NR;
		else if ( nc_max_dt % nr_dt != 0 ) return BLIS_NC_MAX_NONMULTIPLE_OF_NR;
	}

	return BLIS_SUCCESS;
}

err_t bli_check_valid_kc_mod_mult( const blksz_t* kc, const blksz_t* kr )
{
	num_t dt;

	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		dim_t kc_def_dt = bli_blksz_get_def( dt, kc );
		dim_t kc_max_dt = bli_blksz_get_max( dt, kc );
		dim_t kr_dt     = bli_blksz_get_def( dt, kr );

		if      ( kc_def_dt % kr_dt != 0 ) return BLIS_KC_DEF_NONMULTIPLE_OF_KR;
		else if ( kc_max_dt % kr_dt != 0 ) return BLIS_KC_MAX_NONMULTIPLE_OF_KR;
	}

	return BLIS_SUCCESS;
}

err_t bli_check_valid_mr_even( const blksz_t* mr, const mbool_t* row_pref )
{
	num_t dt;

	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		if ( bli_is_complex( dt ) )
			continue;

		dim_t mr_dt        = bli_blksz_get_def( dt, mr );
		dim_t packmr_dt    = bli_blksz_get_max( dt, mr );
		bool  prefers_rows = bli_mbool_get_dt( dt, row_pref );

		if ( prefers_rows )
			continue;

		if      ( mr_dt % 2 != 0 ) return BLIS_MR_NOT_EVEN_FOR_REAL_TYPE;
		else if ( packmr_dt % 2 != 0 ) return BLIS_PACKMR_NOT_EVEN_FOR_REAL_TYPE;
	}

	return BLIS_SUCCESS;
}

err_t bli_check_valid_nr_even( const blksz_t* nr, const mbool_t* row_pref )
{
	num_t dt;

	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		if ( bli_is_complex( dt ) )
			continue;

		dim_t nr_dt = bli_blksz_get_def( dt, nr );
		dim_t packnr_dt = bli_blksz_get_max( dt, nr );
		bool  prefers_rows = bli_mbool_get_dt( dt, row_pref );

		if ( !prefers_rows )
			continue;

		if      ( nr_dt % 2 != 0 ) return BLIS_NR_NOT_EVEN_FOR_REAL_TYPE;
		else if ( packnr_dt % 2 != 0 ) return BLIS_PACKNR_NOT_EVEN_FOR_REAL_TYPE;
	}

	return BLIS_SUCCESS;
}

