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

#ifndef BLIS_PARAM_MACRO_DEFS_H
#define BLIS_PARAM_MACRO_DEFS_H


// -- Parameter query macros --

// buffer

BLIS_INLINE bool bli_is_aligned_to( siz_t p, siz_t size )
{
	return ( bool )
	       ( p % size == 0 );
}

BLIS_INLINE bool bli_is_unaligned_to( siz_t p, siz_t size )
{
	return ( bool )
	       ( p % size != 0 );
}

BLIS_INLINE siz_t bli_offset_past_alignment( siz_t p, siz_t size )
{
	return ( siz_t )
	       ( p % size );
}


// datatype

BLIS_INLINE bool bli_is_float( num_t dt )
{
	return ( bool )
	       ( dt == BLIS_FLOAT );
}

BLIS_INLINE bool bli_is_double( num_t dt )
{
	return ( bool )
	       ( dt == BLIS_DOUBLE );
}

BLIS_INLINE bool bli_is_scomplex( num_t dt )
{
	return ( bool )
	       ( dt == BLIS_SCOMPLEX );
}

BLIS_INLINE bool bli_is_dcomplex( num_t dt )
{
	return ( bool )
	       ( dt == BLIS_DCOMPLEX );
}

BLIS_INLINE bool bli_is_constant( num_t dt )
{
	return ( bool )
	       ( dt == BLIS_CONSTANT );
}

BLIS_INLINE bool bli_is_int( num_t dt )
{
	return ( bool )
	       ( dt == BLIS_INT );
}

BLIS_INLINE bool bli_is_real( num_t dt )
{
	return ( bool )
	       ( bli_is_float( dt ) ||
	                   bli_is_double( dt ) );
}

BLIS_INLINE bool bli_is_complex( num_t dt )
{
	return ( bool )
	       ( bli_is_scomplex( dt ) ||
	                   bli_is_dcomplex( dt ) );
}

BLIS_INLINE bool bli_is_single_prec( num_t dt )
{
	return ( bool )
	       ( bli_is_float( dt ) ||
	                   bli_is_scomplex( dt ) );
}

BLIS_INLINE bool bli_is_double_prec( num_t dt )
{
	return ( bool )
	       ( bli_is_double( dt ) ||
	                   bli_is_dcomplex( dt ) );
}

BLIS_INLINE dom_t bli_dt_domain( num_t dt )
{
	return ( dom_t )
	       ( dt & BLIS_DOMAIN_BIT );
}

BLIS_INLINE bool bli_dt_dom_is_real( num_t dt )
{
	return ( bool )
	       ( ( dt & BLIS_DOMAIN_BIT ) == BLIS_REAL );
}

BLIS_INLINE bool bli_dt_dom_is_complex( num_t dt )
{
	return ( bool )
	       ( ( dt & BLIS_DOMAIN_BIT ) == BLIS_COMPLEX );
}

BLIS_INLINE prec_t bli_dt_prec( num_t dt )
{
	return ( prec_t )
	       ( dt & BLIS_PRECISION_BIT );
}

BLIS_INLINE bool bli_dt_prec_is_single( num_t dt )
{
	return ( bool )
	       ( ( dt & BLIS_PRECISION_BIT ) == BLIS_SINGLE_PREC );
}

BLIS_INLINE bool bli_dt_prec_is_double( num_t dt )
{
	return ( bool )
	       ( ( dt & BLIS_PRECISION_BIT ) == BLIS_DOUBLE_PREC );
}

BLIS_INLINE num_t bli_dt_proj_to_real( num_t dt )
{
	return ( num_t )
	       ( dt & ~BLIS_BITVAL_COMPLEX );
}

BLIS_INLINE num_t bli_dt_proj_to_complex( num_t dt )
{
	return ( num_t )
	       ( dt | BLIS_BITVAL_COMPLEX );
}

BLIS_INLINE num_t bli_dt_proj_to_single_prec( num_t dt )
{
	return ( num_t )
	       ( dt & ~BLIS_BITVAL_DOUBLE_PREC );
}

BLIS_INLINE num_t bli_dt_proj_to_double_prec( num_t dt )
{
	return ( num_t )
	       ( dt | BLIS_BITVAL_DOUBLE_PREC );
}


// trans

BLIS_INLINE bool bli_is_notrans( trans_t trans )
{
	return ( bool )
	       ( trans == BLIS_NO_TRANSPOSE );
}

BLIS_INLINE bool bli_is_trans( trans_t trans )
{
	return ( bool )
	       ( trans == BLIS_TRANSPOSE );
}

BLIS_INLINE bool bli_is_conjnotrans( trans_t trans )
{
	return ( bool )
	       ( trans == BLIS_CONJ_NO_TRANSPOSE );
}

BLIS_INLINE bool bli_is_conjtrans( trans_t trans )
{
	return ( bool )
	       ( trans == BLIS_CONJ_TRANSPOSE );
}

BLIS_INLINE bool bli_does_notrans( trans_t trans )
{
	return ( bool )
	       ( (~trans & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS );
}

BLIS_INLINE bool bli_does_trans( trans_t trans )
{
	return ( bool )
	       ( ( trans & BLIS_TRANS_BIT ) == BLIS_BITVAL_TRANS );
}

BLIS_INLINE bool bli_does_noconj( trans_t trans )
{
	return ( bool )
	       ( (~trans & BLIS_CONJ_BIT ) == BLIS_BITVAL_CONJ );
}

BLIS_INLINE bool bli_does_conj( trans_t trans )
{
	return ( bool )
	       ( ( trans & BLIS_CONJ_BIT ) == BLIS_BITVAL_CONJ );
}

BLIS_INLINE trans_t bli_extract_trans( trans_t trans )
{
	return ( trans_t )
	       ( trans & BLIS_TRANS_BIT );
}

BLIS_INLINE conj_t bli_extract_conj( trans_t trans )
{
	return ( conj_t )
	       ( trans & BLIS_CONJ_BIT );
}

BLIS_INLINE trans_t bli_trans_toggled( trans_t trans )
{
	return ( trans_t )
	       ( trans ^ BLIS_TRANS_BIT );
}

BLIS_INLINE trans_t bli_trans_toggled_conj( trans_t trans )
{
	return ( trans_t )
	       ( trans ^ BLIS_CONJ_BIT );
}

BLIS_INLINE trans_t bli_apply_trans( trans_t transapp, trans_t trans )
{
	return ( trans_t )
	       ( trans ^ transapp );
}

BLIS_INLINE void bli_toggle_trans( trans_t* trans )
{
	*trans = bli_trans_toggled( *trans );
}


// side

BLIS_INLINE bool bli_is_left( side_t side )
{
	return ( bool )
	       ( side == BLIS_LEFT );
}

BLIS_INLINE bool bli_is_right( side_t side )
{
	return ( bool )
	       ( side == BLIS_RIGHT );
}

BLIS_INLINE side_t bli_side_toggled( side_t side )
{
	return ( bli_is_left( side ) ? BLIS_RIGHT : BLIS_LEFT );
}

BLIS_INLINE void bli_toggle_side( side_t* side )
{
	*side = bli_side_toggled( *side );
}


// uplo

BLIS_INLINE bool bli_is_lower( uplo_t uplo )
{
	return ( bool )
	       ( uplo == BLIS_LOWER );
}

BLIS_INLINE bool bli_is_upper( uplo_t uplo )
{
	return ( bool )
	       ( uplo == BLIS_UPPER );
}

BLIS_INLINE bool bli_is_upper_or_lower( uplo_t uplo )
{
	return ( bool )
	       ( bli_is_upper( uplo ) ||
	         bli_is_lower( uplo ) );
}

BLIS_INLINE bool bli_is_dense( uplo_t uplo )
{
	return ( bool )
	       ( uplo == BLIS_DENSE );
}

BLIS_INLINE bool bli_is_zeros( uplo_t uplo )
{
	return ( bool )
	       ( uplo == BLIS_ZEROS );
}

BLIS_INLINE uplo_t bli_uplo_toggled( uplo_t uplo )
{
	return ( uplo_t )
	       ( bli_is_upper_or_lower( uplo )
	         ? ( ( uplo ^ BLIS_LOWER_BIT ) ^ BLIS_UPPER_BIT )
	         :     uplo
	       );
}

BLIS_INLINE void bli_toggle_uplo( uplo_t* uplo )
{
	*uplo = bli_uplo_toggled( *uplo );
}


// structure

BLIS_INLINE bool bli_is_general( struc_t struc )
{
	return ( bool )
	       ( struc == BLIS_GENERAL );
}

BLIS_INLINE bool bli_is_hermitian( struc_t struc )
{
	return ( bool )
	       ( struc == BLIS_HERMITIAN );
}

BLIS_INLINE bool bli_is_symmetric( struc_t struc )
{
	return ( bool )
	       ( struc == BLIS_SYMMETRIC );
}

BLIS_INLINE bool bli_is_triangular( struc_t struc )
{
	return ( bool )
	       ( struc == BLIS_TRIANGULAR );
}

BLIS_INLINE bool bli_is_herm_or_symm( struc_t struc )
{
	return ( bool )
	       ( bli_is_hermitian( struc ) ||
	         bli_is_symmetric( struc ) );
}


// conj

BLIS_INLINE bool bli_is_noconj( conj_t conj )
{
	return ( bool )
	       ( conj == BLIS_NO_CONJUGATE );
}

BLIS_INLINE bool bli_is_conj( conj_t conj )
{
	return ( bool )
	       ( conj == BLIS_CONJUGATE );
}

BLIS_INLINE conj_t bli_conj_toggled( conj_t conj )
{
	return ( conj_t )
	       ( conj ^ BLIS_CONJ_BIT );
}

BLIS_INLINE conj_t bli_apply_conj( conj_t conjapp, conj_t conj )
{
	return ( conj_t )
	       ( conj ^ conjapp );
}

BLIS_INLINE void bli_toggle_conj( conj_t* conj )
{
	*conj = bli_conj_toggled( *conj );
}


// diag

BLIS_INLINE bool bli_is_nonunit_diag( diag_t diag )
{
	return ( bool )
	       ( diag == BLIS_NONUNIT_DIAG );
}

BLIS_INLINE bool bli_is_unit_diag( diag_t diag )
{
	return ( bool )
	       ( diag == BLIS_UNIT_DIAG );
}


// err_t-related

BLIS_INLINE bool bli_is_success( err_t err )
{
	return ( bool )
	       ( err == BLIS_SUCCESS );
}

BLIS_INLINE bool bli_is_failure( err_t err )
{
	return ( bool )
	       ( err != BLIS_SUCCESS );
}


// dimension-related

BLIS_INLINE bool bli_zero_dim1( dim_t m )
{
	return ( bool )
	       ( m == 0 );
}

BLIS_INLINE bool bli_zero_dim2( dim_t m, dim_t n )
{
	return ( bool )
	       ( m == 0 || n == 0 );
}

BLIS_INLINE bool bli_zero_dim3( dim_t m, dim_t n, dim_t k )
{
	return ( bool )
	       ( m == 0 || n == 0 || k == 0 );
}

BLIS_INLINE bool bli_nonzero_dim( dim_t m )
{
	return ( bool )
	       ( m > 0 );
}

BLIS_INLINE bool bli_vector_dim( dim_t m, dim_t n )
{
	return ( bool )
	       ( m == 1 ? n : m );
}

BLIS_INLINE bool bli_is_vector( dim_t m, dim_t n )
{
	return ( bool )
	       ( m == 1 || n == 1 );
}

BLIS_INLINE bool bli_is_row_vector( dim_t m, dim_t n )
{
	return ( bool )
	       ( m == 1 );
}

BLIS_INLINE bool bli_is_col_vector( dim_t m, dim_t n )
{
	return ( bool )
	       ( n == 1 );
}

BLIS_INLINE void bli_set_dim_with_side( side_t side, dim_t m, dim_t n, dim_t* dim )
{
	if ( bli_is_left( side ) ) *dim = m;
	else                       *dim = n;
}

BLIS_INLINE void bli_set_dims_with_trans( trans_t trans, dim_t m, dim_t n, dim_t* mt, dim_t* nt )
{
	if ( bli_does_notrans( trans ) ) { *mt = m; *nt = n; }
	else                             { *mt = n; *nt = m; }
}

BLIS_INLINE void bli_set_dims_incs_with_trans( trans_t trans,
                                          dim_t  m,  dim_t  n,  inc_t  rs,  inc_t  cs,
                                          dim_t* mt, dim_t* nt, inc_t* rst, inc_t* cst )
{
	if ( bli_does_notrans( trans ) ) { *mt = m; *nt = n; *rst = rs; *cst = cs; }
	else                             { *mt = n; *nt = m; *rst = cs; *cst = rs; }
}


// blocksize-related

BLIS_INLINE dim_t bli_determine_blocksize_dim_f( dim_t i, dim_t dim, dim_t b_alg )
{
	return ( dim_t )
	       ( bli_min( b_alg, dim - i ) );
}

BLIS_INLINE dim_t bli_determine_blocksize_dim_b( dim_t i, dim_t dim, dim_t b_alg )
{
	return ( dim_t )
	       ( i == 0 && dim % b_alg != 0 ? dim % b_alg
	                                    : b_alg );
}


// stride-related

BLIS_INLINE inc_t bli_vector_inc( trans_t trans, dim_t m, dim_t n, inc_t rs, inc_t cs )
{
	return ( inc_t )
	       ( bli_does_notrans( trans ) ? ( m == 1 ? cs : rs )
	                                   : ( m == 1 ? rs : cs ) );
}

BLIS_INLINE bool bli_is_row_stored( inc_t rs, inc_t cs )
{
	return ( bool )
	       ( bli_abs( cs ) == 1 );
}

BLIS_INLINE bool bli_is_col_stored( inc_t rs, inc_t cs )
{
	return ( bool )
	       ( bli_abs( rs ) == 1 );
}

BLIS_INLINE bool bli_is_row_stored_f( dim_t m, dim_t n, inc_t rs, inc_t cs )
{
	return ( bool )
	       ( cs == 1 && ( rs > 1 || n == 1 ) );
}

BLIS_INLINE bool bli_is_col_stored_f( dim_t m, dim_t n, inc_t rs, inc_t cs )
{
	return ( bool )
	       ( rs == 1 && ( cs > 1 || m == 1 ) );
}

BLIS_INLINE bool bli_is_gen_stored( inc_t rs, inc_t cs )
{
	return ( bool )
	       ( bli_abs( rs ) != 1 &&
	         bli_abs( cs ) != 1 );
}

BLIS_INLINE bool bli_is_preferentially_stored( inc_t rs, inc_t cs, bool row_pref )
{
	return ( bli_is_row_stored( rs, cs ) &&  row_pref ) ||
	       ( bli_is_col_stored( rs, cs ) && !row_pref );
}

BLIS_INLINE bool bli_is_row_tilted( dim_t m, dim_t n, inc_t rs, inc_t cs )
{
	return ( bool )
	       ( bli_abs( cs ) == bli_abs( rs )
	         ? n < m
	         : bli_abs( cs ) < bli_abs( rs ) );
}

BLIS_INLINE bool bli_is_col_tilted( dim_t m, dim_t n, inc_t rs, inc_t cs )
{
	return ( bool )
	       ( bli_abs( rs ) == bli_abs( cs )
	         ? m < n
	         : bli_abs( rs ) < bli_abs( cs ) );
}

BLIS_INLINE bool bli_has_nonunit_inc1( inc_t s1 )
{
	return ( bool )
	       ( s1 != 1 );
}

BLIS_INLINE bool bli_has_nonunit_inc2( inc_t s1, inc_t s2 )
{
	return ( bool )
	       ( s1 != 1 || s2 != 1 );
}

BLIS_INLINE bool bli_has_nonunit_inc3( inc_t s1, inc_t s2, inc_t s3 )
{
	return ( bool )
	       ( s1 != 1 || s2 != 1 || s3 != 1 );
}


// diag offset-related

BLIS_INLINE void bli_negate_diag_offset( doff_t* diagoff )
{
	*diagoff = -(*diagoff);
}

BLIS_INLINE void bli_shift_diag_offset_to_grow_uplo( uplo_t uplo, doff_t* diagoff )
{
	if      ( bli_is_upper( uplo ) ) *diagoff -= 1;
	else if ( bli_is_lower( uplo ) ) *diagoff += 1;
}

BLIS_INLINE void bli_shift_diag_offset_to_shrink_uplo( uplo_t uplo, doff_t* diagoff )
{
	if      ( bli_is_upper( uplo ) ) *diagoff += 1;
	else if ( bli_is_lower( uplo ) ) *diagoff -= 1;
}

BLIS_INLINE doff_t bli_diag_offset_with_trans( trans_t trans, doff_t diagoff )
{
	return ( doff_t )
	       ( bli_does_trans( trans ) ? -diagoff : diagoff );
}

BLIS_INLINE bool bli_is_strictly_above_diag( doff_t diagoff, trans_t trans, dim_t m, dim_t n )
{
	return ( bool )
	       ( bli_does_trans( trans )
	         ? ( ( doff_t )n <= -diagoff )
	         : ( ( doff_t )m <= -diagoff ) );
}

BLIS_INLINE bool bli_is_strictly_below_diag( doff_t diagoff, trans_t trans, dim_t m, dim_t n )
{
	return ( bool )
	       ( bli_does_trans( trans )
	         ? ( ( doff_t )m <=  diagoff )
	         : ( ( doff_t )n <=  diagoff ) );
}

BLIS_INLINE bool bli_is_outside_diag( doff_t diagoff, trans_t trans, dim_t m, dim_t n )
{
	return ( bool )
	       ( bli_is_strictly_above_diag( diagoff, trans, m, n ) ||
	         bli_is_strictly_below_diag( diagoff, trans, m, n ) );
}

BLIS_INLINE bool bli_is_stored_subpart( doff_t diagoff, trans_t trans, uplo_t uplo, dim_t m, dim_t n )
{
	return ( bool )
	( ( bli_is_upper( uplo ) && bli_is_strictly_above_diag( diagoff, trans, m, n ) ) ||
	  ( bli_is_lower( uplo ) && bli_is_strictly_below_diag( diagoff, trans, m, n ) ) );
}

BLIS_INLINE bool bli_is_unstored_subpart( doff_t diagoff, trans_t trans, uplo_t uplo, dim_t m, dim_t n )
{
	return ( bool )
	( ( bli_is_upper( uplo ) && bli_is_strictly_below_diag( diagoff, trans, m, n ) ) ||
	  ( bli_is_lower( uplo ) && bli_is_strictly_above_diag( diagoff, trans, m, n ) ) );
}

BLIS_INLINE bool bli_is_strictly_above_diag_n( doff_t diagoff, dim_t m, dim_t n )
{
	return ( bool )
	       ( ( doff_t )m <= -diagoff );
}

BLIS_INLINE bool bli_is_strictly_below_diag_n( doff_t diagoff, dim_t m, dim_t n )
{
	return ( bool )
	       ( ( doff_t )n <=  diagoff );
}

BLIS_INLINE bool bli_intersects_diag_n( doff_t diagoff, dim_t m, dim_t n )
{
	return ( bool )
	       ( !bli_is_strictly_above_diag_n( diagoff, m, n ) &&
	         !bli_is_strictly_below_diag_n( diagoff, m, n ) );
}

BLIS_INLINE bool bli_is_outside_diag_n( doff_t diagoff, dim_t m, dim_t n )
{
	return ( bool )
	       ( bli_is_strictly_above_diag_n( diagoff, m, n ) ||
	         bli_is_strictly_below_diag_n( diagoff, m, n ) );
}

BLIS_INLINE bool bli_is_stored_subpart_n( doff_t diagoff, uplo_t uplo, dim_t m, dim_t n )
{
	return ( bool )
	( ( bli_is_upper( uplo ) && bli_is_strictly_above_diag_n( diagoff, m, n ) ) ||
	  ( bli_is_lower( uplo ) && bli_is_strictly_below_diag_n( diagoff, m, n ) ) );
}

BLIS_INLINE bool bli_is_unstored_subpart_n( doff_t diagoff, uplo_t uplo, dim_t m, dim_t n )
{
	return ( bool )
	( ( bli_is_upper( uplo ) && bli_is_strictly_below_diag_n( diagoff, m, n ) ) ||
	  ( bli_is_lower( uplo ) && bli_is_strictly_above_diag_n( diagoff, m, n ) ) );
}


// pruning-related

BLIS_INLINE void bli_prune_unstored_region_top_l( doff_t* diagoff, dim_t* m, dim_t* n, dim_t* offm_inc )
{
	*offm_inc = 0;

	// If the diagonal intersects the left side of the matrix,
	// ignore the area above that intersection.
	if ( *diagoff < 0 )
	{
		*m        = *m + *diagoff;
		*offm_inc =    - *diagoff;
		*diagoff  = 0;
	}
}

BLIS_INLINE void bli_prune_unstored_region_right_l( doff_t* diagoff, dim_t* m, dim_t* n, dim_t* offn_inc )
{
	*offn_inc = 0;

	// If the diagonal intersects the bottom side of the matrix,
	// ignore the area to the right of that intersection.
	if ( *n > *diagoff + *m )
	{
		*n = *diagoff + *m;
	}
}

BLIS_INLINE void bli_prune_unstored_region_left_u( doff_t* diagoff, dim_t* m, dim_t* n, dim_t* offn_inc )
{
	*offn_inc = 0;

	// If the diagonal intersects the top side of the matrix,
	// ignore the area to the left of that intersection.
	if ( *diagoff > 0 )
	{
		*n        = *n - *diagoff;
		*offn_inc =    + *diagoff;
		*diagoff  = 0;
	}
}

BLIS_INLINE void bli_prune_unstored_region_bottom_u( doff_t* diagoff, dim_t* m, dim_t* n, dim_t* offm_inc )
{
	*offm_inc = 0;

	// If the diagonal intersects the right side of the matrix,
	// ignore the area below that intersection.
	if ( *m > -(*diagoff) + *n )
	{
		*m = -(*diagoff) + *n;
	}
}


// thread range-related

BLIS_INLINE void bli_rotate180_trapezoid( doff_t* diagoff, uplo_t* uplo, dim_t* m, dim_t* n )
{
	*diagoff = *n - *diagoff - *m;
	bli_toggle_uplo( uplo );
}

BLIS_INLINE void bli_reflect_about_diag( doff_t* diagoff, uplo_t* uplo, dim_t* m, dim_t* n )
{
	bli_swap_dims( m, n );
	bli_negate_diag_offset( diagoff );
	bli_toggle_uplo( uplo );
}

// we don't know the type of a, so this must be a macro
// rs_a and cs_a must be variables and not expressions
#define bli_reflect_to_stored_part( diagoff, a, rs_a, cs_a ) \
do { \
	a += ( diagoff ) * ( cs_a - rs_a ); \
	bli_swap_incs( &rs_a, &cs_a ); \
} while (0) \

BLIS_INLINE void bli_reverse_index_direction( dim_t n, dim_t* start, dim_t* end )
{
	dim_t start2 = n - *start;
	dim_t end2   = n - *end;
	*start = end2;
	*end   = start2;
}


// mdim_t-related

BLIS_INLINE bool bli_is_m_dim( mdim_t mdim )
{
	return ( bool )
	       ( mdim == BLIS_M );
}

BLIS_INLINE bool bli_is_n_dim( mdim_t mdim )
{
	return ( bool )
	       ( mdim == BLIS_N );
}

BLIS_INLINE mdim_t bli_dim_toggled( mdim_t mdim )
{
	return ( mdim_t )
	       ( mdim == BLIS_M ? BLIS_N : BLIS_M );
}

BLIS_INLINE void bli_toggle_dim( mdim_t* mdim )
{
	*mdim = bli_dim_toggled( *mdim );
}


// stor3_t-related

BLIS_INLINE stor3_t bli_stor3_from_strides( inc_t rs_c, inc_t cs_c,
                                       inc_t rs_a, inc_t cs_a,
                                       inc_t rs_b, inc_t cs_b  )
{
	// If any matrix is general-stored, return the stor3_t id for the
	// general-purpose sup microkernel.
	if ( bli_is_gen_stored( rs_c, cs_c ) ||
	     bli_is_gen_stored( rs_a, cs_a ) ||
	     bli_is_gen_stored( rs_b, cs_b ) ) return BLIS_XXX;

	// Otherwise, compute and return the stor3_t id as follows.
	const bool c_is_col = bli_is_col_stored( rs_c, cs_c );
	const bool a_is_col = bli_is_col_stored( rs_a, cs_a );
	const bool b_is_col = bli_is_col_stored( rs_b, cs_b );

	return ( stor3_t )( 4 * c_is_col +
	                    2 * a_is_col +
	                    1 * b_is_col );
}

BLIS_INLINE stor3_t bli_stor3_trans( stor3_t id )
{
#if 1
	stor3_t map[ BLIS_NUM_3OP_RC_COMBOS ]
	=
	{
	  ( stor3_t )7,  // BLIS_RRR = 0  ->  BLIS_CCC = 7
	  ( stor3_t )5,  // BLIS_RRC = 1  ->  BLIS_CRC = 5
	  ( stor3_t )6,  // BLIS_RCR = 2  ->  BLIS_CCR = 6
	  ( stor3_t )4,  // BLIS_RCC = 3  ->  BLIS_CRR = 4
	  ( stor3_t )3,  // BLIS_CRR = 4  ->  BLIS_RCC = 3
	  ( stor3_t )1,  // BLIS_CRC = 5  ->  BLIS_RRC = 1
	  ( stor3_t )2,  // BLIS_CCR = 6  ->  BLIS_RCR = 2
	  ( stor3_t )0,  // BLIS_CCC = 7  ->  BLIS_RRR = 0
	};

	return map[id];
#else
	return   ( ( id & 0x4 ) ^ 0x4 )        | // flip c bit
	       ( ( ( id & 0x1 ) ^ 0x1 ) << 1 ) | // flip b bit and move to a position
	       ( ( ( id & 0x2 ) ^ 0x2 ) >> 1 );  // flip a bit and move to b position
#endif
}

BLIS_INLINE ukr_t bli_stor3_ukr( stor3_t id )
{
	switch ( id )
	{
		case BLIS_RRR: return BLIS_GEMMSUP_RRR_UKR;
		case BLIS_RRC: return BLIS_GEMMSUP_RRC_UKR;
		case BLIS_RCR: return BLIS_GEMMSUP_RCR_UKR;
		case BLIS_RCC: return BLIS_GEMMSUP_RCC_UKR;
		case BLIS_CRR: return BLIS_GEMMSUP_CRR_UKR;
		case BLIS_CRC: return BLIS_GEMMSUP_CRC_UKR;
		case BLIS_CCR: return BLIS_GEMMSUP_CCR_UKR;
		case BLIS_CCC: return BLIS_GEMMSUP_CCC_UKR;
		default: return BLIS_GEMMSUP_XXX_UKR;
	}
}

BLIS_INLINE stor3_t bli_stor3_transa( stor3_t id )
{
#if 0
	stor3_t map[ BLIS_NUM_3OP_RC_COMBOS ]
	=
	{
	  ( stor3_t )1,  // BLIS_RRR = 0  ->  BLIS_RRC = 1
	  ( stor3_t )0,  // BLIS_RRC = 1  ->  BLIS_RRR = 0
	  ( stor3_t )3,  // BLIS_RCR = 2  ->  BLIS_RCC = 3
	  ( stor3_t )2,  // BLIS_RCC = 3  ->  BLIS_RCR = 2
	  ( stor3_t )5,  // BLIS_CRR = 4  ->  BLIS_CRC = 5
	  ( stor3_t )4,  // BLIS_CRC = 5  ->  BLIS_CRR = 4
	  ( stor3_t )7,  // BLIS_CCR = 6  ->  BLIS_CCC = 7
	  ( stor3_t )6,  // BLIS_CCC = 7  ->  BLIS_CCR = 6
	};

	return map[id];
#else
	return ( stor3_t )( id ^ 0x1 );
#endif
}

BLIS_INLINE stor3_t bli_stor3_transb( stor3_t id )
{
#if 0
	stor3_t map[ BLIS_NUM_3OP_RC_COMBOS ]
	=
	{
	  ( stor3_t )2,  // BLIS_RRR = 0  ->  BLIS_RCR = 2
	  ( stor3_t )3,  // BLIS_RRC = 1  ->  BLIS_RCC = 3
	  ( stor3_t )0,  // BLIS_RCR = 2  ->  BLIS_RRR = 0
	  ( stor3_t )1,  // BLIS_RCC = 3  ->  BLIS_RRC = 1
	  ( stor3_t )6,  // BLIS_CRR = 4  ->  BLIS_CCR = 6
	  ( stor3_t )7,  // BLIS_CRC = 5  ->  BLIS_CCC = 7
	  ( stor3_t )4,  // BLIS_CCR = 6  ->  BLIS_CRR = 4
	  ( stor3_t )5,  // BLIS_CCC = 7  ->  BLIS_CRC = 5
	};

	return map[id];
#else
	return ( stor3_t )( id ^ 0x2 );
#endif
}


// index-related

BLIS_INLINE bool bli_is_edge_f( dim_t i, dim_t n_iter, dim_t n_left )
{
	return ( bool )
	       ( i == n_iter - 1 && n_left != 0 );
}

BLIS_INLINE bool bli_is_not_edge_f( dim_t i, dim_t n_iter, dim_t n_left )
{
	return ( bool )
	       ( i != n_iter - 1 || n_left == 0 );
}

BLIS_INLINE bool bli_is_edge_b( dim_t i, dim_t n_iter, dim_t n_left )
{
	return ( bool )
	       ( i == 0 && n_left != 0 );
}

BLIS_INLINE bool bli_is_not_edge_b( dim_t i, dim_t n_iter, dim_t n_left )
{
	return ( bool )
	       ( i != 0 || n_left == 0 );
}

BLIS_INLINE bool bli_is_last_iter_sl( dim_t i, dim_t end_iter )
{
	return ( bool )
	       ( i == end_iter - 1 );
}

BLIS_INLINE bool bli_is_last_iter_rr( dim_t i, dim_t end_iter, dim_t tid, dim_t nth )
{
	return ( bool )
	       ( i == end_iter - 1 - ( ( end_iter - tid - 1 ) % nth ) );
}

BLIS_INLINE bool bli_is_last_iter_slrr( dim_t i, dim_t end_iter, dim_t tid, dim_t nth )
{
#ifdef BLIS_ENABLE_JRIR_RR
	return bli_is_last_iter_rr( i, end_iter, tid, nth );
#else // ifdef ( _SLAB || _TLB )
	return bli_is_last_iter_sl( i, end_iter );
#endif
}

BLIS_INLINE bool bli_is_last_iter_l( dim_t i, dim_t end_iter, dim_t tid, dim_t nth )
{
	return bli_is_last_iter_slrr( i, end_iter, tid, nth );
}

BLIS_INLINE bool bli_is_last_iter_u( doff_t diagoff, dim_t mr, dim_t nr, inc_t inc )
{
	return bli_is_strictly_below_diag_n( diagoff + inc*mr, mr, nr );
}

BLIS_INLINE bool bli_is_last_iter_tlb_l( dim_t i, dim_t end_iter )
{
	return bli_is_last_iter_sl( i, end_iter );
}

BLIS_INLINE bool bli_is_last_iter_tlb_u( doff_t diagoff, dim_t mr, dim_t nr )
{
	return bli_is_strictly_below_diag_n( diagoff + 1*mr, mr, nr );
}

BLIS_INLINE bool bli_is_my_iter_sl( dim_t i, dim_t st, dim_t en )
{
	return ( st <= i && i < en );
}

BLIS_INLINE bool bli_is_my_iter_rr( dim_t i, dim_t work_id, dim_t n_way )
{
	return ( i % n_way == work_id % n_way );
}

BLIS_INLINE bool bli_is_my_iter( dim_t i, dim_t st, dim_t en, dim_t work_id, dim_t n_way )
{
	// NOTE: This function is (as of this writing) only called from packm.
	// If the structure of the cpp macros below is ever changed, make sure
	// it is still consistent with that of bli_thread_range_slrr() since
	// these functions are used together in packm.

#ifdef BLIS_ENABLE_JRIR_RR
	return bli_is_my_iter_rr( i, work_id, n_way );
#else // ifdef ( _SLAB || _TLB )
	return bli_is_my_iter_sl( i, st, en );
#endif
}


// packbuf_t-related

BLIS_INLINE guint_t bli_packbuf_index( packbuf_t buf_type )
{
	return ( guint_t )
	       ( ( buf_type & BLIS_PACK_BUFFER_BITS ) >> BLIS_PACK_BUFFER_SHIFT );
}

// pack_t-related

BLIS_INLINE bool bli_is_packed( pack_t schema )
{
	return ( bool )
	       ( schema & BLIS_PACK_BIT );
}

BLIS_INLINE bool bli_is_panel_packed( pack_t schema )
{
	return ( bool )
	       ( schema & BLIS_PACK_PANEL_BIT );
}

BLIS_INLINE bool bli_is_1r_packed( pack_t schema )
{
	return ( bool )
	       ( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_1R );
}

BLIS_INLINE bool bli_is_1e_packed( pack_t schema )
{
	return ( bool )
	       ( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_1E );
}

BLIS_INLINE bool bli_is_1m_packed( pack_t schema )
{
	return ( bool )
	       ( bli_is_1r_packed( schema ) ||
	         bli_is_1e_packed( schema ) );
}

BLIS_INLINE bool bli_is_ro_packed( pack_t schema )
{
	return ( bool )
	       ( ( schema & BLIS_PACK_FORMAT_BITS ) == BLIS_BITVAL_RO );
}

BLIS_INLINE bool bli_is_nat_packed( pack_t schema )
{
	return ( bool )
	       ( ( schema & BLIS_PACK_FORMAT_BITS ) == 0 );
}

BLIS_INLINE bool bli_is_ind_packed( pack_t schema )
{
	return ( bool )
	       ( ( schema & BLIS_PACK_FORMAT_BITS ) != 0 );
}

BLIS_INLINE guint_t bli_pack_schema_index( pack_t schema )
{
	return ( guint_t )
	       ( ( schema & BLIS_PACK_FORMAT_BITS ) >> BLIS_PACK_FORMAT_SHIFT );
}


// Set dimensions, increments, effective uplo/diagoff, etc for ONE matrix
// argument.

BLIS_INLINE void bli_set_dims_incs_uplo_1m
     (
       doff_t  diagoffa, diag_t diaga,
       uplo_t  uploa,    dim_t  m,          dim_t  n,      inc_t  rs_a, inc_t  cs_a,
       uplo_t* uplo_eff, dim_t* n_elem_max, dim_t* n_iter, inc_t* inca, inc_t* lda,
       dim_t*  ij0,      dim_t* n_shift
     )
{
	// This is to prevent the compiler from warning about uninitialized
	// variables.
	*ij0     = 0;
	*n_shift = 0;

	// If matrix A is entirely "unstored", that is, if either:
	// - A is lower-stored and entirely above the diagonal, or
	// - A is upper-stored and entirely below the diagonal
	// then we mark the storage as implicitly zero.
	if ( bli_is_unstored_subpart( diagoffa, BLIS_NO_TRANSPOSE, uploa, m, n ) )
	{
		*uplo_eff = BLIS_ZEROS;
	}
	else
	{
		doff_t diagoffa_use_ = diagoffa;
		doff_t diagoff_eff_;
		dim_t  n_iter_max_;

		if ( bli_is_unit_diag( diaga ) )
			bli_shift_diag_offset_to_shrink_uplo( uploa, &diagoffa_use_ );

		// If matrix A is entirely "stored", that is, if either:
		// - A is upper-stored and entirely above the diagonal, or
		// - A is lower-stored and entirely below the diagonal
		// then we mark the storage as dense.
		if ( bli_is_stored_subpart( diagoffa_use_, BLIS_NO_TRANSPOSE, uploa, m, n ) )
			uploa = BLIS_DENSE;

		n_iter_max_  = n;
		*n_elem_max   = m;
		*inca         = rs_a;
		*lda          = cs_a;
		*uplo_eff     = uploa;
		diagoff_eff_ = diagoffa_use_;

		if ( bli_is_row_tilted( *n_elem_max, n_iter_max_, *inca, *lda ) )
		{
			bli_swap_dims( &n_iter_max_, n_elem_max );
			bli_swap_incs( inca, lda );
			bli_toggle_uplo( uplo_eff );
			bli_negate_diag_offset( &diagoff_eff_ );
		}

		if ( bli_is_dense( *uplo_eff ) )
		{
			*n_iter = n_iter_max_;
		}
		else if ( bli_is_upper( *uplo_eff ) )
		{
			if ( diagoff_eff_ < 0 )
			{
				*ij0        = 0;
				*n_shift    = -diagoff_eff_;
				*n_elem_max = bli_min( *n_elem_max, *n_shift + bli_min( m, n ) );
				*n_iter     = n_iter_max_;
			}
			else
			{
				*ij0        = diagoff_eff_;
				*n_shift    = 0;
				*n_iter     = n_iter_max_ - diagoff_eff_;
			}
		}
		else // if ( bli_is_lower( *uplo_eff ) )
		{
			if ( diagoff_eff_ < 0 )
			{
				*ij0        = -diagoff_eff_;
				*n_shift    = 0;
				*n_elem_max = *n_elem_max + diagoff_eff_;
				*n_iter     = bli_min( *n_elem_max, bli_min( m, n ) );
			}
			else
			{
				*ij0        = 0;
				*n_shift    = diagoff_eff_;
				*n_iter     = bli_min( n_iter_max_, *n_shift + bli_min( m, n ) );
			}
		}
	}
}

// Set dimensions, increments, effective uplo/diagoff, etc for ONE matrix
// argument (without column-wise stride optimization).

BLIS_INLINE void bli_set_dims_incs_uplo_1m_noswap
     (
       doff_t  diagoffa, diag_t diaga,
       uplo_t  uploa,    dim_t  m,          dim_t  n,      inc_t  rs_a, inc_t  cs_a,
       uplo_t* uplo_eff, dim_t* n_elem_max, dim_t* n_iter, inc_t* inca, inc_t* lda,
       dim_t*  ij0,      dim_t* n_shift
     )
{
	// This is to prevent the compiler from warning about uninitialized
	// variables.
	*ij0     = 0;
	*n_shift = 0;

	// If matrix A is entirely "unstored", that is, if either:
	// - A is lower-stored and entirely above the diagonal, or
	// - A is upper-stored and entirely below the diagonal
	// then we mark the storage as implicitly zero.
	if ( bli_is_unstored_subpart( diagoffa, BLIS_NO_TRANSPOSE, uploa, m, n ) )
	{
		*uplo_eff = BLIS_ZEROS;
	}
	else
	{
		doff_t diagoffa_use_ = diagoffa;
		doff_t diagoff_eff_;
		dim_t  n_iter_max_;

		if ( bli_is_unit_diag( diaga ) )
			bli_shift_diag_offset_to_shrink_uplo( uploa, &diagoffa_use_ );

		// If matrix A is entirely "stored", that is, if either:
		// - A is upper-stored and entirely above the diagonal, or
		// - A is lower-stored and entirely below the diagonal
		// then we mark the storage as dense.
		if ( bli_is_stored_subpart( diagoffa_use_, BLIS_NO_TRANSPOSE, uploa, m, n ) )
			uploa = BLIS_DENSE;

		n_iter_max_  = n;
		*n_elem_max   = m;
		*inca         = rs_a;
		*lda          = cs_a;
		*uplo_eff     = uploa;
		diagoff_eff_ = diagoffa_use_;

		if ( bli_is_dense( *uplo_eff ) )
		{
			*n_iter = n_iter_max_;
		}
		else if ( bli_is_upper( *uplo_eff ) )
		{
			if ( diagoff_eff_ < 0 )
			{
				*ij0        = 0;
				*n_shift    = -diagoff_eff_;
				*n_elem_max = bli_min( *n_elem_max, *n_shift + bli_min( m, n ) );
				*n_iter     = n_iter_max_;
			}
			else
			{
				*ij0        = diagoff_eff_;
				*n_shift    = 0;
				*n_iter     = n_iter_max_ - diagoff_eff_;
			}
		}
		else // if ( bli_is_lower( *uplo_eff ) )
		{
			if ( diagoff_eff_ < 0 )
			{
				*ij0        = -diagoff_eff_;
				*n_shift    = 0;
				*n_elem_max = *n_elem_max + diagoff_eff_;
				*n_iter     = bli_min( *n_elem_max, bli_min( m, n ) );
			}
			else
			{
				*ij0        = 0;
				*n_shift    = diagoff_eff_;
				*n_iter     = bli_min( n_iter_max_, *n_shift + bli_min( m, n ) );
			}
		}
	}
}

// Set dimensions and increments for TWO matrix arguments.

BLIS_INLINE void bli_set_dims_incs_2m
     (
       trans_t transa,
       dim_t  m,      dim_t  n,      inc_t  rs_a, inc_t  cs_a,
                                     inc_t  rs_b, inc_t  cs_b,
       dim_t* n_elem, dim_t* n_iter, inc_t* inca, inc_t* lda,
                                     inc_t* incb, inc_t* ldb
     )
{
	{
		*n_iter = n;
		*n_elem = m;
		*inca   = rs_a;
		*lda    = cs_a;
		*incb   = rs_b;
		*ldb    = cs_b;

		if ( bli_does_trans( transa ) )
		{
			bli_swap_incs( inca, lda );
		}

		if ( bli_is_row_tilted( *n_elem, *n_iter, *incb, *ldb ) &&
		     bli_is_row_tilted( *n_elem, *n_iter, *inca, *lda ) )
		{
			bli_swap_dims( n_iter, n_elem );
			bli_swap_incs( inca, lda );
			bli_swap_incs( incb, ldb );
		}
	}
}

// Set dimensions, increments, effective uplo/diagoff, etc for TWO matrix
// arguments.

BLIS_INLINE void bli_set_dims_incs_uplo_2m
     (
       doff_t  diagoffa, diag_t diaga, trans_t transa,
       uplo_t  uploa,    dim_t  m,          dim_t  n,      inc_t  rs_a, inc_t  cs_a,
                                                           inc_t  rs_b, inc_t  cs_b,
       uplo_t* uplo_eff, dim_t* n_elem_max, dim_t* n_iter, inc_t* inca, inc_t* lda,
                                                           inc_t* incb, inc_t* ldb,
       dim_t*  ij0,      dim_t* n_shift
     )
{
	// This is to prevent the compiler from warning about uninitialized
	// variables.
	*ij0     = 0;
	*n_shift = 0;

	// If matrix A is entirely "unstored", that is, if either:
	// - A is lower-stored and entirely above the diagonal, or
	// - A is upper-stored and entirely below the diagonal
	// then we mark the storage as implicitly zero.
	if ( bli_is_unstored_subpart( diagoffa, transa, uploa, m, n ) )
	{
		*uplo_eff = BLIS_ZEROS;
	}
	else
	{
		doff_t diagoffa_use_ = diagoffa;
		doff_t diagoff_eff_;
		dim_t  n_iter_max_;

		if ( bli_is_unit_diag( diaga ) )
			bli_shift_diag_offset_to_shrink_uplo( uploa, &diagoffa_use_ );

		// If matrix A is entirely "stored", that is, if either:
		// - A is upper-stored and entirely above the diagonal, or
		// - A is lower-stored and entirely below the diagonal
		// then we mark the storage as dense.
		if ( bli_is_stored_subpart( diagoffa_use_, transa, uploa, m, n ) )
			uploa = BLIS_DENSE;

		n_iter_max_   = n;
		*n_elem_max   = m;
		*inca         = rs_a;
		*lda          = cs_a;
		*incb         = rs_b;
		*ldb          = cs_b;
		*uplo_eff     = uploa;
		diagoff_eff_ = diagoffa_use_;

		if ( bli_does_trans( transa ) )
		{
			bli_swap_incs( inca, lda );
			bli_toggle_uplo( uplo_eff );
			bli_negate_diag_offset( &diagoff_eff_ );
		}

		if ( bli_is_row_tilted( *n_elem_max, n_iter_max_, *incb, *ldb ) &&
		     bli_is_row_tilted( *n_elem_max, n_iter_max_, *inca, *lda ) )
		{
			bli_swap_dims( &n_iter_max_, n_elem_max );
			bli_swap_incs( inca, lda );
			bli_swap_incs( incb, ldb );
			bli_toggle_uplo( uplo_eff );
			bli_negate_diag_offset( &diagoff_eff_ );
		}

		if ( bli_is_dense( *uplo_eff ) )
		{
			*n_iter = n_iter_max_;
		}
		else if ( bli_is_upper( *uplo_eff ) )
		{
			if ( diagoff_eff_ < 0 )
			{
				*ij0        = 0;
				*n_shift    = -diagoff_eff_;
				*n_elem_max = bli_min( *n_elem_max, *n_shift + bli_min( m, n ) );
				*n_iter     = n_iter_max_;
			}
			else
			{
				*ij0        = diagoff_eff_;
				*n_shift    = 0;
				*n_iter     = n_iter_max_ - diagoff_eff_;
			}
		}
		else // if ( bli_is_lower( *uplo_eff ) )
		{
			if ( diagoff_eff_ < 0 )
			{
				*ij0        = -diagoff_eff_;
				*n_shift    = 0;
				*n_elem_max = *n_elem_max + diagoff_eff_;
				*n_iter     = bli_min( *n_elem_max, bli_min( m, n ) );
			}
			else
			{
				*ij0        = 0;
				*n_shift    = diagoff_eff_;
				*n_iter     = bli_min( n_iter_max_, *n_shift + bli_min( m, n ) );
			}
		}
	}
}

// Set dimensions, increments, etc for ONE matrix argument when operating
// on the diagonal.

BLIS_INLINE void bli_set_dims_incs_1d
     (
       doff_t diagoffx,
       dim_t  m,    dim_t  n,      inc_t  rs_x, inc_t  cs_x,
       dim_t* offx, dim_t* n_elem, inc_t* incx
     )
{
	if ( diagoffx < 0 )
	{
		*n_elem = bli_min( m - ( dim_t )(-diagoffx), n );
		*offx   = ( dim_t )(-diagoffx) * rs_x;
	}
	else
	{
		*n_elem = bli_min( n - ( dim_t )( diagoffx), m );
		*offx   = ( dim_t )( diagoffx) * cs_x;
	}

	*incx = rs_x + cs_x; \
}

// Set dimensions, increments, etc for TWO matrix arguments when operating
// on diagonals.
BLIS_INLINE void bli_set_dims_incs_2d
     (
       doff_t diagoffx, trans_t transx,
       dim_t  m, dim_t  n, inc_t  rs_x, inc_t  cs_x,
                           inc_t  rs_y, inc_t  cs_y,
       dim_t* offx, dim_t* offy, dim_t* n_elem,
       inc_t* incx, inc_t* incy
     )
{
	doff_t diagoffy_ = bli_diag_offset_with_trans( transx, diagoffx );

	if ( diagoffx < 0 ) *offx = -diagoffx * rs_x;
	else                *offx =  diagoffx * cs_x;

	if ( diagoffy_ < 0 )
	{
		*n_elem = bli_min( m - ( dim_t )(-diagoffy_), n );
		*offy   = -diagoffy_ * rs_y;
	}
	else
	{
		*n_elem = bli_min( n - ( dim_t )( diagoffy_), m );
		*offy   = diagoffy_ * cs_y;
	}

	*incx = rs_x + cs_x;
	*incy = rs_y + cs_y;
}


#endif
