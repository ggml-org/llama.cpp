/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

#ifndef BLIS_OBJ_MACRO_DEFS_H
#define BLIS_OBJ_MACRO_DEFS_H


// -- Object query/modification macros --

// Info query

BLIS_INLINE num_t bli_obj_dt( const obj_t* obj )
{
	return ( num_t )
	       ( obj->info & BLIS_DATATYPE_BITS );
}

BLIS_INLINE bool bli_obj_is_float( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_dt( obj ) == BLIS_BITVAL_FLOAT_TYPE );
}

BLIS_INLINE bool bli_obj_is_double( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_dt( obj ) == BLIS_BITVAL_DOUBLE_TYPE );
}

BLIS_INLINE bool bli_obj_is_scomplex( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_dt( obj ) == BLIS_BITVAL_SCOMPLEX_TYPE );
}

BLIS_INLINE bool bli_obj_is_dcomplex( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_dt( obj ) == BLIS_BITVAL_DCOMPLEX_TYPE );
}

BLIS_INLINE bool bli_obj_is_int( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_dt( obj ) == BLIS_BITVAL_INT_TYPE );
}

BLIS_INLINE bool bli_obj_is_const( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_dt( obj ) == BLIS_BITVAL_CONST_TYPE );
}

BLIS_INLINE dom_t bli_obj_domain( const obj_t* obj )
{
	return ( dom_t )
	       ( obj->info & BLIS_DOMAIN_BIT );
}

BLIS_INLINE prec_t bli_obj_prec( const obj_t* obj )
{
	return ( prec_t )
	       ( obj->info & BLIS_PRECISION_BIT );
}

BLIS_INLINE bool bli_obj_is_single_prec( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_prec( obj ) == BLIS_BITVAL_SINGLE_PREC );
}

BLIS_INLINE bool bli_obj_is_double_prec( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_prec( obj ) == BLIS_BITVAL_DOUBLE_PREC );
}

BLIS_INLINE num_t bli_obj_dt_proj_to_single_prec( const obj_t* obj )
{
	return ( num_t )
	       ( bli_obj_dt( obj ) & ~BLIS_BITVAL_SINGLE_PREC );
}

BLIS_INLINE num_t bli_obj_dt_proj_to_double_prec( const obj_t* obj )
{
	return ( num_t )
	       ( bli_obj_dt( obj ) | BLIS_BITVAL_DOUBLE_PREC );
}

BLIS_INLINE bool bli_obj_is_real( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_domain( obj ) == BLIS_BITVAL_REAL &&
	         !bli_obj_is_const( obj ) );
}

BLIS_INLINE bool bli_obj_is_complex( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_domain( obj ) == BLIS_BITVAL_COMPLEX &&
	         !bli_obj_is_const( obj ) );
}

BLIS_INLINE num_t bli_obj_dt_proj_to_real( const obj_t* obj )
{
	return ( num_t )
	       ( bli_obj_dt( obj ) & ~BLIS_BITVAL_COMPLEX );
}

BLIS_INLINE num_t bli_obj_dt_proj_to_complex( const obj_t* obj )
{
	return ( num_t )
	       ( bli_obj_dt( obj ) | BLIS_BITVAL_COMPLEX );
}

BLIS_INLINE prec_t bli_obj_comp_prec( const obj_t* obj )
{
	return ( prec_t )
	       ( ( obj->info & BLIS_COMP_PREC_BIT ) >> BLIS_COMP_PREC_SHIFT );
}

// NOTE: This function queries info2.
BLIS_INLINE num_t bli_obj_scalar_dt( const obj_t* obj )
{
	return ( num_t )
	       ( ( obj->info2 & BLIS_SCALAR_DT_BITS ) >> BLIS_SCALAR_DT_SHIFT );
}

// NOTE: This function queries info2.
BLIS_INLINE dom_t bli_obj_scalar_domain( const obj_t* obj )
{
	return ( dom_t )
	       ( ( obj->info2 & BLIS_SCALAR_DOMAIN_BIT ) >> BLIS_SCALAR_DT_SHIFT );
}

// NOTE: This function queries info2.
BLIS_INLINE prec_t bli_obj_scalar_prec( const obj_t* obj )
{
	return ( prec_t )
	       ( ( obj->info2 & BLIS_SCALAR_PREC_BIT ) >> BLIS_SCALAR_DT_SHIFT );
}

BLIS_INLINE trans_t bli_obj_conjtrans_status( const obj_t* obj )
{
	return ( trans_t )
	       ( obj->info & BLIS_CONJTRANS_BITS );
}

BLIS_INLINE trans_t bli_obj_onlytrans_status( const obj_t* obj )
{
	return ( trans_t )
	       ( obj->info & BLIS_TRANS_BIT );
}

BLIS_INLINE bool bli_obj_has_trans( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_onlytrans_status( obj ) == BLIS_BITVAL_TRANS );
}

BLIS_INLINE bool bli_obj_has_notrans( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_onlytrans_status( obj ) == BLIS_BITVAL_NO_TRANS );
}

BLIS_INLINE conj_t bli_obj_conj_status( const obj_t* obj )
{
	return ( conj_t )
	       ( obj->info & BLIS_CONJ_BIT );
}

BLIS_INLINE bool bli_obj_has_conj( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_conj_status( obj ) == BLIS_BITVAL_CONJ );
}

BLIS_INLINE bool bli_obj_has_noconj( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_conj_status( obj ) == BLIS_BITVAL_NO_CONJ );
}

BLIS_INLINE uplo_t bli_obj_uplo( const obj_t* obj )
{
	return ( uplo_t )
	       ( obj->info & BLIS_UPLO_BITS );
}

BLIS_INLINE bool bli_obj_is_upper( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_uplo( obj ) == BLIS_BITVAL_UPPER );
}

BLIS_INLINE bool bli_obj_is_lower( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_uplo( obj ) == BLIS_BITVAL_LOWER );
}

BLIS_INLINE bool bli_obj_is_upper_or_lower( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_upper( obj ) ||
	         bli_obj_is_lower( obj ) );
}

BLIS_INLINE bool bli_obj_is_dense( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_uplo( obj ) == BLIS_BITVAL_DENSE );
}

BLIS_INLINE bool bli_obj_is_zeros( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_uplo( obj ) == BLIS_BITVAL_ZEROS );
}

BLIS_INLINE diag_t bli_obj_diag( const obj_t* obj )
{
	return ( diag_t )
	       ( obj->info & BLIS_UNIT_DIAG_BIT );
}

BLIS_INLINE bool bli_obj_has_nonunit_diag( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_diag( obj ) == BLIS_BITVAL_NONUNIT_DIAG );
}

BLIS_INLINE bool bli_obj_has_unit_diag( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_diag( obj ) == BLIS_BITVAL_UNIT_DIAG );
}

BLIS_INLINE bool bli_obj_has_inverted_diag( const obj_t* obj )
{
	return ( bool )
	       ( ( obj->info & BLIS_INVERT_DIAG_BIT ) == BLIS_BITVAL_INVERT_DIAG );
}

BLIS_INLINE bool bli_obj_is_pack_rev_if_upper( const obj_t* obj )
{
	return ( bool )
	       ( ( obj->info & BLIS_PACK_REV_IF_UPPER_BIT ) == BLIS_BITVAL_PACK_REV_IF_UPPER );
}

BLIS_INLINE bool bli_obj_is_pack_rev_if_lower( const obj_t* obj )
{
	return ( bool )
	       ( ( obj->info & BLIS_PACK_REV_IF_LOWER_BIT ) == BLIS_BITVAL_PACK_REV_IF_LOWER );
}

BLIS_INLINE pack_t bli_obj_pack_schema( const obj_t* obj )
{
	return ( pack_t )
	       ( obj->info & BLIS_PACK_SCHEMA_BITS );
}

BLIS_INLINE bool bli_obj_is_packed( const obj_t* obj )
{
	return ( bool )
	       ( obj->info & BLIS_PACK_BIT );
}

BLIS_INLINE bool bli_obj_is_panel_packed( const obj_t* obj )
{
	return ( bool )
	       ( obj->info & BLIS_PACK_PANEL_BIT );
}

BLIS_INLINE packbuf_t bli_obj_pack_buffer_type( const obj_t* obj )
{
	return ( packbuf_t )
	       ( obj->info & BLIS_PACK_BUFFER_BITS );
}

BLIS_INLINE struc_t bli_obj_struc( const obj_t* obj )
{
	return ( struc_t )
	       ( obj->info & BLIS_STRUC_BITS );
}

BLIS_INLINE bool bli_obj_is_general( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_struc( obj ) == BLIS_BITVAL_GENERAL );
}

BLIS_INLINE bool bli_obj_is_hermitian( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_struc( obj ) == BLIS_BITVAL_HERMITIAN );
}

BLIS_INLINE bool bli_obj_is_symmetric( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_struc( obj ) == BLIS_BITVAL_SYMMETRIC );
}

BLIS_INLINE bool bli_obj_is_triangular( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_struc( obj ) == BLIS_BITVAL_TRIANGULAR );
}

// Info modification

BLIS_INLINE void bli_obj_apply_trans( trans_t trans, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( obj->info ^ trans );
}

BLIS_INLINE void bli_obj_apply_conj( conj_t conj, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( obj->info ^ conj );
}

BLIS_INLINE void bli_obj_set_conjtrans( trans_t trans, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_CONJTRANS_BITS ) | trans );
}

BLIS_INLINE void bli_obj_set_onlytrans( trans_t trans, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_TRANS_BIT ) | trans );
}

BLIS_INLINE void bli_obj_set_conj( conj_t conj, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_CONJ_BIT ) | conj );
}

BLIS_INLINE void bli_obj_set_uplo( uplo_t uplo, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_UPLO_BITS ) | uplo );
}

BLIS_INLINE void bli_obj_set_diag( diag_t diag, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_UNIT_DIAG_BIT ) | diag );
}

BLIS_INLINE void bli_obj_set_invert_diag( invdiag_t invdiag, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_INVERT_DIAG_BIT ) | invdiag );
}

BLIS_INLINE void bli_obj_set_dt( num_t dt, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_DATATYPE_BITS ) | dt );
}

BLIS_INLINE void bli_obj_set_comp_prec( prec_t dt, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_COMP_PREC_BIT ) |
	              ( dt << BLIS_COMP_PREC_SHIFT ) );
}

// NOTE: This function queries and modifies info2.
BLIS_INLINE void bli_obj_set_scalar_dt( num_t dt, obj_t* obj )
{
	obj->info2 = ( objbits_t )
	             ( ( obj->info2 & ~BLIS_SCALAR_DT_BITS ) |
	               ( dt << BLIS_SCALAR_DT_SHIFT ) );
}

// NOTE: This function queries and modifies info2.
BLIS_INLINE void bli_obj_set_scalar_domain( dom_t dt, obj_t* obj )
{
	obj->info2 = ( objbits_t )
	             ( ( obj->info2 & ~BLIS_SCALAR_DOMAIN_BIT ) |
	               ( dt << BLIS_SCALAR_DT_SHIFT ) );
}

// NOTE: This function queries and modifies info2.
BLIS_INLINE void bli_obj_set_scalar_prec( prec_t dt, obj_t* obj )
{
	obj->info2 = ( objbits_t )
	             ( ( obj->info2 & ~BLIS_SCALAR_PREC_BIT ) |
	               ( dt << BLIS_SCALAR_DT_SHIFT ) );
}

BLIS_INLINE void bli_obj_set_pack_schema( pack_t schema, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_PACK_SCHEMA_BITS ) | schema );
}

BLIS_INLINE void bli_obj_set_pack_order_if_upper( packord_t ordif, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_PACK_REV_IF_UPPER_BIT ) | ordif );
}

BLIS_INLINE void bli_obj_set_pack_order_if_lower( packord_t ordif, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_PACK_REV_IF_LOWER_BIT ) | ordif );
}

// NOTE: The packbuf_t bitfield in the obj_t is currently unused. Instead,
// packbuf_t is stored/used from the context in order to support various
// induced methods. (Though ideally the packbuf_t field would only be
// present in the control tree).
BLIS_INLINE void bli_obj_set_pack_buffer_type( packbuf_t buf_type, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_PACK_BUFFER_BITS ) | buf_type );
}

BLIS_INLINE void bli_obj_set_struc( struc_t struc, obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( ( obj->info & ~BLIS_STRUC_BITS ) | struc );
}

BLIS_INLINE void bli_obj_toggle_trans( obj_t* obj )
{
	bli_obj_apply_trans( BLIS_TRANSPOSE, obj );
}

BLIS_INLINE void bli_obj_toggle_conj( obj_t* obj )
{
	bli_obj_apply_conj( BLIS_CONJUGATE, obj );
}

BLIS_INLINE void bli_obj_toggle_uplo( obj_t* obj )
{
	obj->info = ( objbits_t )
	            ( obj->info ^ BLIS_LOWER_BIT ) ^ BLIS_UPPER_BIT;
}

// Root matrix query

BLIS_INLINE obj_t* bli_obj_root( const obj_t* obj )
{
	return ( obj_t* )( obj->root );
}

BLIS_INLINE bool bli_obj_root_is_general( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_general( bli_obj_root( obj ) ) );
}

BLIS_INLINE bool bli_obj_root_is_hermitian( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_hermitian( bli_obj_root( obj ) ) );
}

BLIS_INLINE bool bli_obj_root_is_symmetric( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_symmetric( bli_obj_root( obj ) ) );
}

BLIS_INLINE bool bli_obj_root_is_triangular( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_triangular( bli_obj_root( obj ) ) );
}

BLIS_INLINE bool bli_obj_root_is_herm_or_symm( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_hermitian( bli_obj_root( obj ) ) ||
	         bli_obj_is_symmetric( bli_obj_root( obj ) ) );
}

BLIS_INLINE bool bli_obj_root_is_upper( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_upper( bli_obj_root( obj ) ) );
}

BLIS_INLINE bool bli_obj_root_is_lower( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_lower( bli_obj_root( obj ) ) );
}

// Root matrix modification

BLIS_INLINE void bli_obj_set_as_root( obj_t* obj )
{
	obj->root = obj;
}

// Diagonal offset query

BLIS_INLINE doff_t bli_obj_diag_offset( const obj_t* obj )
{
	return ( doff_t )
	       ( obj->diag_off );
}

BLIS_INLINE doff_t bli_obj_diag_offset_after_trans( const obj_t* obj )
{
	return ( doff_t )
	       ( bli_obj_has_trans( obj ) ? -bli_obj_diag_offset( obj )
	                                  :  bli_obj_diag_offset( obj ) );
}

// Diagonal offset modification

BLIS_INLINE void bli_obj_set_diag_offset( doff_t offset, obj_t* obj )
{
	obj->diag_off = ( doff_t )offset;
}

BLIS_INLINE void bli_obj_negate_diag_offset( obj_t* obj )
{
	obj->diag_off = -(obj->diag_off);
}

BLIS_INLINE void bli_obj_inc_diag_offset( doff_t offset, obj_t* obj )
{
	obj->diag_off += ( doff_t )offset;
}

// Dimension query

BLIS_INLINE dim_t bli_obj_length( const obj_t* obj )
{
	return ( obj->dim[ BLIS_M ] );
}

BLIS_INLINE dim_t bli_obj_width( const obj_t* obj )
{
	return ( obj->dim[ BLIS_N ] );
}

BLIS_INLINE dim_t bli_obj_dim( mdim_t mdim, const obj_t* obj )
{
	return ( obj->dim[ mdim ] );
}

BLIS_INLINE dim_t bli_obj_min_dim( const obj_t* obj )
{
	return bli_min( bli_obj_length( obj ),
	                bli_obj_width( obj ) );
}

BLIS_INLINE dim_t bli_obj_max_dim( const obj_t* obj )
{
	return bli_max( bli_obj_length( obj ),
	                bli_obj_width( obj ) );
}

BLIS_INLINE dim_t bli_obj_length_after_trans( const obj_t* obj )
{
	return ( bli_obj_has_trans( obj ) ? bli_obj_width( obj )
	                                  : bli_obj_length( obj ) );
}

BLIS_INLINE dim_t bli_obj_width_after_trans( const obj_t* obj )
{
	return ( bli_obj_has_trans( obj ) ? bli_obj_length( obj )
	                                  : bli_obj_width( obj ) );
}

BLIS_INLINE bool bli_obj_is_1x1( const obj_t* x )
{
	return ( bool )
	       ( bli_obj_length( x ) == 1 &&
	         bli_obj_width( x ) == 1 );
}

// Stride/increment query

BLIS_INLINE inc_t bli_obj_row_stride( const obj_t* obj )
{
	return ( obj->rs );
}

BLIS_INLINE inc_t bli_obj_col_stride( const obj_t* obj )
{
	return ( obj->cs );
}

BLIS_INLINE inc_t bli_obj_imag_stride( const obj_t* obj )
{
	return ( obj->is );
}

BLIS_INLINE inc_t bli_obj_row_stride_mag( const obj_t* obj )
{
	return ( inc_t )
	       ( bli_abs( obj->rs ) );
}

BLIS_INLINE inc_t bli_obj_col_stride_mag( const obj_t* obj )
{
	return ( inc_t )
	       ( bli_abs( obj->cs ) );
}

BLIS_INLINE inc_t bli_obj_imag_stride_mag( const obj_t* obj )
{
	return ( inc_t )
	       ( bli_abs( obj->is ) );
}

// Note: The purpose of these functions is to obtain the length and width
// of the smallest submatrices of an object that could still encompass
// the stored data above (if obj is upper) or below (if obj is lower)
// the diagonal.
BLIS_INLINE dim_t bli_obj_length_stored( const obj_t* obj )
{
	return ( dim_t )
	       ( bli_obj_is_upper( obj )
	         ? bli_min( bli_obj_length( obj ),
	                    bli_obj_width( obj )  - bli_obj_diag_offset( obj ) )
	         : bli_min( bli_obj_length( obj ),
	                    bli_obj_length( obj ) + bli_obj_diag_offset( obj ) )
	       );
}

BLIS_INLINE dim_t bli_obj_width_stored( const obj_t* obj )
{
	return ( dim_t )
	       ( bli_obj_is_lower( obj )
	         ? bli_min( bli_obj_width( obj ),
	                    bli_obj_length( obj ) + bli_obj_diag_offset( obj ) )
	         : bli_min( bli_obj_width( obj ),
	                    bli_obj_width( obj )  - bli_obj_diag_offset( obj ) )
	       );
}

BLIS_INLINE dim_t bli_obj_length_stored_after_trans( const obj_t* obj )
{
	return ( bli_obj_has_trans( obj ) ? bli_obj_width_stored( obj )
	                                  : bli_obj_length_stored( obj ) );
}

BLIS_INLINE dim_t bli_obj_width_stored_after_trans( const obj_t* obj )
{
	return ( bli_obj_has_trans( obj ) ? bli_obj_length_stored( obj )
	                                  : bli_obj_width_stored( obj ) );
}

BLIS_INLINE dim_t bli_obj_vector_dim( const obj_t* x )
{
	return ( bli_obj_length( x ) == 1 ? bli_obj_width( x )
	                                  : bli_obj_length( x ) );
}

BLIS_INLINE inc_t bli_obj_vector_inc( const obj_t* x )
{
	return ( bli_obj_is_1x1( x ) ? 1 :
	         ( bli_obj_length( x ) == 1 ? bli_obj_col_stride( x )
	                                    : bli_obj_row_stride( x ) )
	       );
}

BLIS_INLINE bool bli_obj_is_vector( const obj_t* x )
{
	return ( bool )
	       ( bli_obj_length( x ) == 1 ||
	         bli_obj_width(  x ) == 1 );
}

BLIS_INLINE bool bli_obj_is_row_vector( const obj_t* x )
{
	return ( bool )
	       ( bli_obj_length( x ) == 1 );
}

BLIS_INLINE bool bli_obj_is_col_vector( const obj_t* x )
{
	return ( bool )
	       ( bli_obj_width( x ) == 1 );
}

BLIS_INLINE bool bli_obj_has_zero_dim( const obj_t* x )
{
	return ( bool )
	       ( bli_obj_length( x ) == 0 ||
	         bli_obj_width(  x ) == 0 );
}

// Dimension modification

BLIS_INLINE void bli_obj_set_length( dim_t m, obj_t* obj )
{
	obj->dim[ BLIS_M ] = m;
}

BLIS_INLINE void bli_obj_set_width( dim_t n, obj_t* obj )
{
	obj->dim[ BLIS_N ] = n;
}

BLIS_INLINE void bli_obj_set_dim( mdim_t mdim, dim_t dim_val, obj_t* obj )
{
	obj->dim[ mdim ] = dim_val;
}

BLIS_INLINE void bli_obj_set_dims( dim_t m, dim_t n, obj_t* obj )
{
	bli_obj_set_length( m, obj );
	bli_obj_set_width( n, obj );
}

BLIS_INLINE void bli_obj_set_dims_with_trans( trans_t trans, dim_t m, dim_t n, obj_t* obj )
{
	if ( bli_does_notrans( trans ) )
	{
		bli_obj_set_length( m, obj );
		bli_obj_set_width( n, obj );
	}
	else // if ( bli_does_trans( trans ) )
	{
		bli_obj_set_length( n, obj );
		bli_obj_set_width( m, obj );
	}
}

// Stride/increment predicates

//
// NOTE: The following two macros differ from their non-obj counterparts
// in that they do not identify m x 1 and 1 x n objects as row-stored and
// column-stored, respectively, which is needed when considering packed
// objects. But this is okay, since none of the invocations of these
// "obj" macros are used on packed matrices.
//

BLIS_INLINE bool bli_obj_is_row_stored( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_col_stride_mag( obj ) == 1 );
}

BLIS_INLINE bool bli_obj_is_col_stored( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_row_stride_mag( obj ) == 1 );
}

BLIS_INLINE bool bli_obj_is_gen_stored( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_row_stride_mag( obj ) != 1 &&
	         bli_obj_col_stride_mag( obj ) != 1 );
}

BLIS_INLINE bool bli_obj_is_row_tilted( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_col_stride_mag( obj ) < bli_obj_row_stride_mag( obj ) );
}

BLIS_INLINE bool bli_obj_is_col_tilted( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_row_stride_mag( obj ) < bli_obj_col_stride_mag( obj ) );
}

// Stride/increment modification

BLIS_INLINE void bli_obj_set_row_stride( inc_t rs, obj_t* obj )
{
	obj->rs = rs;
}

BLIS_INLINE void bli_obj_set_col_stride( inc_t cs, obj_t* obj )
{
	obj->cs = cs;
}

BLIS_INLINE void bli_obj_set_strides( inc_t rs, inc_t cs, obj_t* obj )
{
	bli_obj_set_row_stride( rs, obj );
	bli_obj_set_col_stride( cs, obj );
}

BLIS_INLINE void bli_obj_set_imag_stride( inc_t is, obj_t* obj )
{
	obj->is = is;
}

// Offset query

BLIS_INLINE dim_t bli_obj_row_off( const obj_t* obj )
{
	return ( obj->off[ BLIS_M ] );
}

BLIS_INLINE dim_t bli_obj_col_off( const obj_t* obj )
{
	return ( obj->off[ BLIS_N ] );
}

BLIS_INLINE dim_t bli_obj_off( mdim_t mdim, const obj_t* obj )
{
	return ( obj->off[ mdim ] );
}

// Offset modification

BLIS_INLINE void bli_obj_set_off( mdim_t mdim, dim_t offset, obj_t* obj )
{
	obj->off[ mdim ] = offset;
}

BLIS_INLINE void bli_obj_set_offs( dim_t offm, dim_t offn, obj_t* obj )
{
	bli_obj_set_off( BLIS_M, offm, obj );
	bli_obj_set_off( BLIS_N, offn, obj );
}

BLIS_INLINE void bli_obj_inc_off( mdim_t mdim, dim_t offset, obj_t* obj )
{
	obj->off[ mdim ] += offset;
}

BLIS_INLINE void bli_obj_inc_offs( dim_t offm, dim_t offn, obj_t* obj )
{
	bli_obj_inc_off( BLIS_M, offm, obj );
	bli_obj_inc_off( BLIS_N, offn, obj );
}

// Diagonal offset predicates

BLIS_INLINE bool bli_obj_is_strictly_above_diag( const obj_t* obj )
{
	return ( bool )
	       ( ( doff_t )bli_obj_length( obj ) <= -bli_obj_diag_offset( obj ) );
}

BLIS_INLINE bool bli_obj_is_strictly_below_diag( const obj_t* obj )
{
	return ( bool )
	       ( ( doff_t )bli_obj_width( obj ) <= bli_obj_diag_offset( obj ) );
}

BLIS_INLINE bool bli_obj_is_outside_diag( const obj_t* obj )
{
	return ( bool )
	       ( bli_obj_is_strictly_above_diag( obj ) ||
	         bli_obj_is_strictly_below_diag( obj ) );
}

BLIS_INLINE bool bli_obj_intersects_diag( const obj_t* obj )
{
	return ( bool )
	       ( !bli_obj_is_strictly_above_diag( obj ) &&
	         !bli_obj_is_strictly_below_diag( obj ) );
}

BLIS_INLINE bool bli_obj_is_unstored_subpart( const obj_t* obj )
{
	return ( bool )
	       ( ( bli_obj_root_is_lower( obj ) && bli_obj_is_strictly_above_diag( obj ) ) ||
	         ( bli_obj_root_is_upper( obj ) && bli_obj_is_strictly_below_diag( obj ) ) );
}

// Buffer address query

BLIS_INLINE void* bli_obj_buffer( const obj_t* obj )
{
	return ( void* )
	       ( obj->buffer );
}

// Buffer address modification

BLIS_INLINE void bli_obj_set_buffer( void* p, obj_t* obj )
{
	obj->buffer = p;
}

// Bufferless scalar field query

BLIS_INLINE void* bli_obj_internal_scalar_buffer( const obj_t* obj )
{
	return ( void* )
	       ( &( obj->scalar ) );
}

// Bufferless scalar field modification

BLIS_INLINE void bli_obj_copy_internal_scalar( const obj_t* a, obj_t* b )
{
	b->scalar = a->scalar;
}

// Element size query

BLIS_INLINE siz_t bli_obj_elem_size( const obj_t* obj )
{
	return ( siz_t )
	       ( obj->elem_size );
}

// Element size modification

BLIS_INLINE void bli_obj_set_elem_size( siz_t size, obj_t* obj )
{
	obj->elem_size = size;
}

// Packed matrix info query

BLIS_INLINE dim_t bli_obj_padded_length( const obj_t* obj )
{
	return ( obj->m_padded );
}

BLIS_INLINE dim_t bli_obj_padded_width( const obj_t* obj )
{
	return ( obj->n_padded );
}

// Packed matrix info modification

BLIS_INLINE void bli_obj_set_padded_length( dim_t m, obj_t* obj )
{
	obj->m_padded = m;
}

BLIS_INLINE void bli_obj_set_padded_width( dim_t n, obj_t* obj )
{
	obj->n_padded = n;
}

BLIS_INLINE void bli_obj_set_padded_dims( dim_t m, dim_t n, obj_t* obj )
{
	bli_obj_set_padded_length( m, obj );
	bli_obj_set_padded_width( n, obj );
}

// Packed panel info query

BLIS_INLINE dim_t bli_obj_panel_length( const obj_t* obj )
{
	return ( obj->m_panel );
}

BLIS_INLINE dim_t bli_obj_panel_width( const obj_t* obj )
{
	return ( obj->n_panel );
}

BLIS_INLINE inc_t bli_obj_panel_dim( const obj_t* obj )
{
	return ( obj->pd );
}

BLIS_INLINE inc_t bli_obj_panel_stride( const obj_t* obj )
{
	return ( obj->ps );
}

// Packed panel info modification

BLIS_INLINE void bli_obj_set_panel_length( dim_t m, obj_t* obj )
{
	obj->m_panel = m;
}

BLIS_INLINE void bli_obj_set_panel_width( dim_t n, obj_t* obj )
{
	obj->n_panel = n;
}

BLIS_INLINE void bli_obj_set_panel_dims( dim_t m, dim_t n, obj_t* obj )
{
	bli_obj_set_panel_length( m, obj );
	bli_obj_set_panel_width( n, obj );
}

BLIS_INLINE void bli_obj_set_panel_dim( inc_t pd, obj_t* obj )
{
	obj->pd = pd;
}

BLIS_INLINE void bli_obj_set_panel_stride( inc_t ps, obj_t* obj )
{
	obj->ps = ps;
}

// stor3_t-related

BLIS_INLINE stor3_t bli_obj_stor3_from_strides( const obj_t* c, const obj_t* a, const obj_t* b )
{
	const inc_t rs_c = bli_obj_row_stride( c );
	const inc_t cs_c = bli_obj_col_stride( c );

	inc_t rs_a, cs_a;
	inc_t rs_b, cs_b;

	if ( bli_obj_has_notrans( a ) )
	{
		rs_a = bli_obj_row_stride( a );
		cs_a = bli_obj_col_stride( a );
	}
	else
	{
		rs_a = bli_obj_col_stride( a );
		cs_a = bli_obj_row_stride( a );
	}

	if ( bli_obj_has_notrans( b ) )
	{
		rs_b = bli_obj_row_stride( b );
		cs_b = bli_obj_col_stride( b );
	}
	else
	{
		rs_b = bli_obj_col_stride( b );
		cs_b = bli_obj_row_stride( b );
	}

	return bli_stor3_from_strides( rs_c, cs_c,
	                               rs_a, cs_a,
	                               rs_b, cs_b  );
}


// -- Initialization-related macros --

// Finish the initialization started by the matrix-specific static initializer
// (e.g. BLIS_OBJECT_INITIALIZER)
// NOTE: This is intended only for use in the BLAS compatibility API and typed
// BLIS API.

BLIS_INLINE void bli_obj_init_finish( num_t dt, dim_t m, dim_t n, void* p, inc_t rs, inc_t cs, obj_t* obj )
{
	bli_obj_set_as_root( obj );

	bli_obj_set_dt( dt, obj );
	bli_obj_set_comp_prec( bli_dt_prec( dt ), obj );

	bli_obj_set_dims( m, n, obj );
	bli_obj_set_strides( rs, cs, obj );

	siz_t elem_size = sizeof( float );
	if ( bli_dt_prec_is_double( dt ) ) elem_size *= 2;
	if ( bli_dt_dom_is_complex( dt ) ) elem_size *= 2;
	bli_obj_set_elem_size( elem_size, obj );

	bli_obj_set_buffer( p, obj );

	bli_obj_set_scalar_dt( dt, obj );
	void* s = bli_obj_internal_scalar_buffer( obj );

	if      ( bli_dt_prec_is_single( dt ) ) { bli_creal( *( scomplex* )s ) = 1.0F;
	                                          bli_cimag( *( scomplex* )s ) = 0.0F; }
	else if ( bli_dt_prec_is_double( dt ) ) { bli_zreal( *( dcomplex* )s ) = 1.0;
	                                          bli_zimag( *( dcomplex* )s ) = 0.0; }
}

// Finish the initialization started by the 1x1-specific static initializer
// (e.g. BLIS_OBJECT_INITIALIZER_1X1)
// NOTE: This is intended only for use in the BLAS compatibility API and typed
// BLIS API.

BLIS_INLINE void bli_obj_init_finish_1x1( num_t dt, void* p, obj_t* obj )
{
	bli_obj_set_as_root( obj );

	bli_obj_set_dt( dt, obj );

	bli_obj_set_buffer( p, obj );
}

// -- Miscellaneous object macros --

// Toggle the region referenced (or "stored").

BLIS_INLINE void bli_obj_toggle_region_ref( obj_t* obj )
{
	if      ( bli_obj_is_upper( obj ) ) bli_obj_inc_diag_offset( -1, obj );
	else if ( bli_obj_is_lower( obj ) ) bli_obj_inc_diag_offset(  1, obj );

	bli_obj_toggle_uplo( obj );
}

BLIS_INLINE void bli_obj_toggle_uplo_if_trans( trans_t trans, obj_t* obj )
{
	if ( bli_does_trans( trans ) &&
	     bli_obj_is_upper_or_lower( obj ) )
	{
		bli_obj_toggle_uplo( obj );
		bli_obj_negate_diag_offset( obj );
	}
}

// Initialize object with default properties (info field).

BLIS_INLINE void bli_obj_set_defaults( obj_t* obj )
{
	obj->info = 0x0;
	obj->info = obj->info | BLIS_BITVAL_DENSE | BLIS_BITVAL_GENERAL;
}

// Acquire buffer at object's submatrix offset (offset-aware buffer query).

BLIS_INLINE void* bli_obj_buffer_at_off( const obj_t* obj )
{
	return ( void* )
	       (
	         ( ( char* )( bli_obj_buffer   ( obj ) ) +
	           ( dim_t )( bli_obj_elem_size( obj ) ) *
	                      ( bli_obj_col_off( obj ) * bli_obj_col_stride( obj ) +
	                        bli_obj_row_off( obj ) * bli_obj_row_stride( obj )
	                      )
	         )
	       );
}

// Acquire buffer from BLIS_CONSTANT object.

BLIS_INLINE const void* bli_obj_buffer_for_const( num_t dt, const obj_t* obj )
{
	void* p;

	if      ( dt == BLIS_FLOAT    ) p = &((( constdata_t* )bli_obj_buffer( obj ))->s);
	else if ( dt == BLIS_DOUBLE   ) p = &((( constdata_t* )bli_obj_buffer( obj ))->d);
	else if ( dt == BLIS_SCOMPLEX ) p = &((( constdata_t* )bli_obj_buffer( obj ))->c);
	else if ( dt == BLIS_DCOMPLEX ) p = &((( constdata_t* )bli_obj_buffer( obj ))->z);
	else                            p = &((( constdata_t* )bli_obj_buffer( obj ))->i);

	return p;
}

// Acquire buffer from scalar (1x1) object, including BLIS_CONSTANT objects.

BLIS_INLINE void* bli_obj_buffer_for_1x1( num_t dt, const obj_t* obj )
{
	return ( void* )
	       ( bli_obj_is_const( obj ) ? bli_obj_buffer_for_const( dt, obj )
	                                 : bli_obj_buffer_at_off( obj )
	       );
}

// Adjust the pointer based on current offsets, zero the offsets, and then
// set the current object as the root. For obj_t's with at least one non-zero
// offset, this effectively makes the obj_t "forget" that it was ever a view
// into a larger matrix.

BLIS_INLINE void bli_obj_reset_origin( obj_t* obj )
{
	bli_obj_set_buffer( bli_obj_buffer_at_off( obj ), obj );
	bli_obj_set_offs( 0, 0, obj );
	bli_obj_set_as_root( obj );
}

// Make a full alias (shallow copy).

BLIS_INLINE void bli_obj_alias_to( const obj_t* a, obj_t* b )
{
	bli_obj_init_full_shallow_copy_of( a, b );
}

// Check if two objects are aliases of one another.

BLIS_INLINE bool bli_obj_is_alias_of( const obj_t* a, const obj_t* b )
{
	return ( bool )
	       ( bli_obj_buffer( a ) == bli_obj_buffer( b ) );
}


// Create an alias with a trans value applied.
// (Note: trans may include a conj component.)

BLIS_INLINE void bli_obj_alias_with_trans( trans_t trans, const obj_t* a, obj_t* b )
{
	bli_obj_alias_to( a, b );
	bli_obj_apply_trans( trans, b );
}

// Create an alias with a conj value applied.

BLIS_INLINE void bli_obj_alias_with_conj( conj_t conja, const obj_t* a, obj_t* b )
{
	bli_obj_alias_to( a, b );
	bli_obj_apply_conj( conja, b );
}

// Alias only the real part.

BLIS_INLINE void bli_obj_real_part( const obj_t* c, obj_t* r )
{
	bli_obj_alias_to( c, r );

	if ( bli_obj_is_complex( c ) )
	{
		// Change the datatypes.
		bli_obj_set_dt( bli_obj_dt_proj_to_real( c ), r );

		// Don't touch the attached scalar datatype.

		// Update the element size.
		siz_t es_c = bli_obj_elem_size( c );
		bli_obj_set_elem_size( es_c/2, r );

		// Update the strides.
		inc_t rs_c = bli_obj_row_stride( c );
		inc_t cs_c = bli_obj_col_stride( c );
		bli_obj_set_strides( 2*rs_c, 2*cs_c, r );

		// Buffer is left unchanged.
	}
}

// Alias only the imaginary part.

BLIS_INLINE void bli_obj_imag_part( const obj_t* c, obj_t* i )
{
	if ( bli_obj_is_complex( c ) )
	{
		bli_obj_alias_to( c, i );

		// Change the datatype.
		bli_obj_set_dt( bli_obj_dt_proj_to_real( c ), i );

		// Don't touch the attached scalar datatype.

		// Update the element size.
		siz_t es_c = bli_obj_elem_size( c );
		bli_obj_set_elem_size( es_c/2, i );

		// Update the strides.
		inc_t rs_c = bli_obj_row_stride( c );
		inc_t cs_c = bli_obj_col_stride( c );
		bli_obj_set_strides( 2*rs_c, 2*cs_c, i );

		// Update the buffer.
		inc_t is_c = bli_obj_imag_stride( c );
		char* p    = ( char* )bli_obj_buffer( c );
		bli_obj_set_buffer( p + is_c * es_c/2, i );
	}
}

// Given a 1x1 object, acquire an address to the buffer depending on whether
// the object is a BLIS_CONSTANT, and also set a datatype associated with the
// chosen buffer (possibly using an auxiliary datatype if the object is
// BLIS_CONSTANT).

BLIS_INLINE void bli_obj_scalar_set_dt_buffer( const obj_t* obj, num_t dt_aux, num_t* dt, void** buf )
{
	if ( bli_obj_is_const( obj ) )
	{
		*dt  = dt_aux;
		*buf = bli_obj_buffer_for_1x1( dt_aux, obj );
	}
	else
	{
		*dt  = bli_obj_dt( obj );
		*buf = bli_obj_buffer_at_off( obj );
	}
}

// Swap all object fields (metadata/properties).

BLIS_INLINE void bli_obj_swap( obj_t* a, obj_t* b )
{
	bool a_root_is_self = ( bli_obj_root( a ) == a );
	bool b_root_is_self = ( bli_obj_root( b ) == b );

	obj_t t = *b; *b = *a; *a = t;

	if ( a_root_is_self ) bli_obj_set_as_root( b );
	if ( b_root_is_self ) bli_obj_set_as_root( a );
}

// Induce a transposition on an object: swap dimensions, increments, and
// offsets, then clear the trans bit.

BLIS_INLINE void bli_obj_induce_trans( obj_t* obj )
{
	// Induce transposition among basic fields.
	dim_t  m        = bli_obj_length( obj );
	dim_t  n        = bli_obj_width( obj );
	inc_t  rs       = bli_obj_row_stride( obj );
	inc_t  cs       = bli_obj_col_stride( obj );
	dim_t  offm     = bli_obj_row_off( obj );
	dim_t  offn     = bli_obj_col_off( obj );
	doff_t diag_off = bli_obj_diag_offset( obj );

	bli_obj_set_dims( n, m, obj );
	bli_obj_set_strides( cs, rs, obj );
	bli_obj_set_offs( offn, offm, obj );
	bli_obj_set_diag_offset( -diag_off, obj );

	if ( bli_obj_is_upper_or_lower( obj ) )
		bli_obj_toggle_uplo( obj );

	// Induce transposition among packed fields.
	dim_t  m_padded = bli_obj_padded_length( obj );
	dim_t  n_padded = bli_obj_padded_width( obj );
	dim_t  m_panel  = bli_obj_panel_length( obj );
	dim_t  n_panel  = bli_obj_panel_width( obj );

	bli_obj_set_padded_dims( n_padded, m_padded, obj );
	bli_obj_set_panel_dims( n_panel, m_panel, obj );

	// Note that this macro DOES NOT touch the transposition bit! If
	// the calling code is using this function to handle an object whose
	// transposition bit is set prior to computation, that code needs
	// to manually clear or toggle the bit, via
	// bli_obj_set_onlytrans() or bli_obj_toggle_trans(),
	// respectively.
}

BLIS_INLINE void bli_obj_induce_fast_trans( obj_t* obj )
{
	// NOTE: This function is only used in situations where the matrices
	// are guaranteed to not have structure or be packed.

	// Induce transposition among basic fields.
	dim_t  m        = bli_obj_length( obj );
	dim_t  n        = bli_obj_width( obj );
	inc_t  rs       = bli_obj_row_stride( obj );
	inc_t  cs       = bli_obj_col_stride( obj );
	dim_t  offm     = bli_obj_row_off( obj );
	dim_t  offn     = bli_obj_col_off( obj );

	bli_obj_set_dims( n, m, obj );
	bli_obj_set_strides( cs, rs, obj );
	bli_obj_set_offs( offn, offm, obj );

	// Note that this macro DOES NOT touch the transposition bit! If
	// the calling code is using this function to handle an object whose
	// transposition bit is set prior to computation, that code needs
	// to manually clear or toggle the bit, via
	// bli_obj_set_onlytrans() or bli_obj_toggle_trans(),
	// respectively.
}

// Sometimes we need to "reflect" a partition because the data we want is
// actually stored on the other side of the diagonal. The nuts and bolts of
// this macro look a lot like an induced transposition, except that the row
// and column strides are left unchanged (which, of course, drastically
// changes the effect of the macro).

BLIS_INLINE void bli_obj_reflect_about_diag( obj_t* obj )
{
	dim_t  m        = bli_obj_length( obj );
	dim_t  n        = bli_obj_width( obj );
	dim_t  offm     = bli_obj_row_off( obj );
	dim_t  offn     = bli_obj_col_off( obj );
	doff_t diag_off = bli_obj_diag_offset( obj );

	bli_obj_set_dims( n, m, obj );
	bli_obj_set_offs( offn, offm, obj );
	bli_obj_set_diag_offset( -diag_off, obj );

	bli_obj_toggle_trans( obj );
}

// Create an alias which refers to only a portion of the original matrix
// without any "historical baggage", stripping out all offsets, transposes,
// and references to the original root object.

BLIS_INLINE void bli_obj_alias_submatrix( const obj_t* a, obj_t* b )
{
	bli_obj_alias_to( a, b );
    bli_obj_reset_origin( b );
    if ( bli_obj_has_trans( b ) )
    {
        bli_obj_induce_trans( b );
        bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, b );
    }
}

#endif
