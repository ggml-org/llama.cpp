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


// -- Matrix partitioning ------------------------------------------------------


void bli_acquire_mpart
     (
             dim_t  i,
             dim_t  j,
             dim_t  bm,
             dim_t  bn,
       const obj_t* parent,
             obj_t* child
     )
{
	// Query the dimensions of the parent object.
	const dim_t m_par = bli_obj_length( parent );
	const dim_t n_par = bli_obj_width( parent );

	// If either i or j is already beyond what exists of the parent matrix,
	// slide them back to the outer dimensions. (What will happen in this
	// scenario is that bm and bn and/or will be reduced to zero so that the
	// child matrix does not refer to anything beyond the bounds of the
	// parent. (Note: This is a safety measure and generally should never
	// be needed if the caller is passing in sane arguments.)
	if ( i > m_par ) i = m_par;
	if ( j > n_par ) j = n_par;

	// If either bm or bn spills out over the edge of the parent matrix,
	// reduce them so that the child matrix fits within the bounds of the
	// parent. (Note: This is a safety measure and generally should never
	// be needed if the caller is passing in sane arguments, though this
	// code is somewhat more likely to be needed than the code above.)
	if ( bm > m_par - i ) bm = m_par - i;
	if ( bn > n_par - j ) bn = n_par - j;

	// Alias the parent object's contents into the child object.
	bli_obj_alias_to( parent, child );

	// Set the offsets and dimensions of the child object. Note that we
	// increment, rather than overwrite, the offsets of the child object
	// in case the parent object already had non-zero offsets (usually
	// because the parent was itself a child a larger grandparent object).
	bli_obj_inc_offs( i, j, child );
	bli_obj_set_dims( bm, bn, child );
}


void bli_acquire_mpart_t2b
     (
             subpart_t req_part,
             dim_t     i,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	bli_acquire_mpart_mdim( BLIS_FWD, req_part, i, b, obj, sub_obj );
}


void bli_acquire_mpart_b2t
     (
             subpart_t req_part,
             dim_t     i,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	bli_acquire_mpart_mdim( BLIS_BWD, req_part, i, b, obj, sub_obj );
}


void bli_acquire_mpart_mdim
     (
             dir_t     direct,
             subpart_t req_part,
             dim_t     i,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	dim_t  m;
	dim_t  n;
	dim_t  m_part   = 0;
	dim_t  n_part   = 0;
	inc_t  offm_inc = 0;
	inc_t  offn_inc = 0;


	// Call a special function for partitioning packed objects. (By only
	// catching those objects packed to panels, we omit cases where the
	// object is packed to row or column storage, as such objects can be
	// partitioned through normally.) Note that the function called below
	// assumes forward partitioning.
	if ( bli_obj_is_panel_packed( obj ) )
		bli_check_error_code( BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_PART );


	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_acquire_mpart_t2b_check( req_part, i, b, obj, sub_obj );


	// Query the m and n dimensions of the object (accounting for
	// transposition, if indicated).
	if ( bli_obj_has_notrans( obj ) )
	{
		m = bli_obj_length( obj );
		n = bli_obj_width( obj );
	}
	else // if ( bli_obj_has_trans( obj ) )
	{
		m = bli_obj_width( obj );
		n = bli_obj_length( obj );
	}


	// Foolproofing: do not let b exceed what's left of the m dimension at
	// row offset i.
	if ( b > m - i ) b = m - i;


	// NOTE: Most of this function implicitly assumes moving forward.
	// When moving backward, we have to relocate i.
	if ( direct == BLIS_BWD )
	{
		// Modify i to account for the fact that we are moving backwards.
		i = m - i - b;
	}


	// Support SUBPART1B (behind SUBPART1) and SUBPART1A (ahead of SUBPART1),
	// to refer to subpartitions 0 and 2 when moving forward, and 2 and 0 when
	// moving backward.
	subpart_t subpart0_alias;
	subpart_t subpart2_alias;

	if ( direct == BLIS_FWD ) { subpart0_alias = BLIS_SUBPART1B;
	                            subpart2_alias = BLIS_SUBPART1A; }
	else                      { subpart0_alias = BLIS_SUBPART1A;
	                            subpart2_alias = BLIS_SUBPART1B; }


	// Compute offset increments and dimensions based on which
	// subpartition is being requested, assuming no transposition.
	if      ( req_part == BLIS_SUBPART0 ||
	          req_part == subpart0_alias )
	{
		// A0 (offm,offn) unchanged.
		// A0 is i x n.
		offm_inc = 0;
		offn_inc = 0;
		m_part   = i;
		n_part   = n;
	}
	else if ( req_part == BLIS_SUBPART1AND0 )
	{
		// A1+A0 (offm,offn) unchanged.
		// A1+A0 is (i+b) x n.
		offm_inc = 0;
		offn_inc = 0;
		m_part   = i + b;
		n_part   = n;
	}
	else if ( req_part == BLIS_SUBPART1 )
	{
		// A1 (offm,offn) += (i,0).
		// A1 is b x n.
		offm_inc = i;
		offn_inc = 0;
		m_part   = b;
		n_part   = n;
	}
	else if ( req_part == BLIS_SUBPART1AND2 )
	{
		// A1+A2 (offm,offn) += (i,0).
		// A1+A2 is (m-i) x n.
		offm_inc = i;
		offn_inc = 0;
		m_part   = m - i;
		n_part   = n;
	}
	else if ( req_part == BLIS_SUBPART2 ||
	          req_part == subpart2_alias )
	{
		// A2 (offm,offn) += (i+b,0).
		// A2 is (m-i-b) x n.
		offm_inc = i + b;
		offn_inc = 0;
		m_part   = m - i - b;
		n_part   = n;
	}


	// Compute the diagonal offset based on the m and n offsets.
	doff_t diagoff_inc = ( doff_t )offm_inc - ( doff_t )offn_inc;


	// Begin by copying the info, elem size, buffer, row stride, and column
	// stride fields of the parent object. Note that this omits copying view
	// information because the new partition will have its own dimensions
	// and offsets.
	bli_obj_init_subpart_from( obj, sub_obj );


	// Modify offsets and dimensions of requested partition based on
	// whether it needs to be transposed.
	if ( bli_obj_has_notrans( obj ) )
	{
		bli_obj_set_dims( m_part, n_part, sub_obj );
		bli_obj_inc_offs( offm_inc, offn_inc, sub_obj );
		bli_obj_inc_diag_offset( diagoff_inc, sub_obj );
	}
	else // if ( bli_obj_has_trans( obj ) )
	{
		bli_obj_set_dims( n_part, m_part, sub_obj );
		bli_obj_inc_offs( offn_inc, offm_inc, sub_obj );
		bli_obj_inc_diag_offset( -diagoff_inc, sub_obj );
	}


	// If the root matrix is not general (ie: has structure defined by the
	// diagonal), and the subpartition does not intersect the root matrix's
	// diagonal, then set the subpartition structure to "general"; otherwise
	// we let the subpartition inherit the storage structure of its immediate
	// parent.
	if ( !bli_obj_root_is_general( sub_obj ) &&
	      bli_obj_is_outside_diag( sub_obj ) )
	{
		// NOTE: This comment may be out-of-date since we now distinguish
		// between uplo properties for the current and root objects...
		// Note that we cannot mark the subpartition object as general/dense
		// here since it makes sense to preserve the existing uplo information
		// a while longer so that the correct kernels are invoked. (Example:
		// incremental packing/computing in gemmt produces subpartitions that
		// appear general/dense, but their uplo fields are needed to be either
		// lower or upper, to determine which macro-kernel gets called in the
		// gemmt_int() back-end.)

		// If the subpartition lies entirely in an "unstored" triangle of the
		// root matrix, then we need to tweak the subpartition. If the root
		// matrix is Hermitian or symmetric, then we reflect the partition to
		// the other side of the diagonal, toggling the transposition bit (and
		// conjugation bit if the root matrix is Hermitian). Or, if the root
		// matrix is triangular, the subpartition should be marked as zero.
		if ( bli_obj_is_unstored_subpart( sub_obj ) )
		{
			if ( bli_obj_root_is_hermitian( sub_obj ) )
			{
				bli_obj_reflect_about_diag( sub_obj );
				bli_obj_toggle_conj( sub_obj );
			}
			else if ( bli_obj_root_is_symmetric( sub_obj ) )
			{
				bli_obj_reflect_about_diag( sub_obj );
			}
			else if ( bli_obj_root_is_triangular( sub_obj ) )
			{
				bli_obj_set_uplo( BLIS_ZEROS, sub_obj );
			}
		}
	}
}


void bli_acquire_mpart_l2r
     (
             subpart_t req_part,
             dim_t     j,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	bli_acquire_mpart_ndim( BLIS_FWD, req_part, j, b, obj, sub_obj );
}


void bli_acquire_mpart_r2l
     (
             subpart_t req_part,
             dim_t     j,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	bli_acquire_mpart_ndim( BLIS_BWD, req_part, j, b, obj, sub_obj );
}


void bli_acquire_mpart_ndim
     (
             dir_t     direct,
             subpart_t req_part,
             dim_t     j,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	dim_t  m;
	dim_t  n;
	dim_t  m_part   = 0;
	dim_t  n_part   = 0;
	inc_t  offm_inc = 0;
	inc_t  offn_inc = 0;


	// Call a special function for partitioning packed objects. (By only
	// catching those objects packed to panels, we omit cases where the
	// object is packed to row or column storage, as such objects can be
	// partitioned through normally.) Note that the function called below
	// assumes forward partitioning.
	if ( bli_obj_is_panel_packed( obj ) )
		bli_check_error_code( BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_PART );


	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_acquire_mpart_l2r_check( req_part, j, b, obj, sub_obj );


	// Query the m and n dimensions of the object (accounting for
	// transposition, if indicated).
	if ( bli_obj_has_notrans( obj ) )
	{
		m = bli_obj_length( obj );
		n = bli_obj_width( obj );
	}
	else // if ( bli_obj_has_trans( obj ) )
	{
		m = bli_obj_width( obj );
		n = bli_obj_length( obj );
	}


	// Foolproofing: do not let b exceed what's left of the n dimension at
	// column offset j.
	if ( b > n - j ) b = n - j;


	// NOTE: Most of this function implicitly assumes moving forward.
	// When moving backward, we have to relocate j.
	if ( direct == BLIS_BWD )
	{
		// Modify j to account for the fact that we are moving backwards.
		j = n - j - b;
	}


	// Support SUBPART1B (behind SUBPART1) and SUBPART1A (ahead of SUBPART1),
	// to refer to subpartitions 0 and 2 when moving forward, and 2 and 0 when
	// moving backward.
	subpart_t subpart0_alias;
	subpart_t subpart2_alias;

	if ( direct == BLIS_FWD ) { subpart0_alias = BLIS_SUBPART1B;
	                            subpart2_alias = BLIS_SUBPART1A; }
	else                      { subpart0_alias = BLIS_SUBPART1A;
	                            subpart2_alias = BLIS_SUBPART1B; }


	// Compute offset increments and dimensions based on which
	// subpartition is being requested, assuming no transposition.
	if      ( req_part == BLIS_SUBPART0 ||
	          req_part == subpart0_alias )
	{
		// A0 (offm,offn) unchanged.
		// A0 is m x j.
		offm_inc = 0;
		offn_inc = 0;
		m_part   = m;
		n_part   = j;
	}
	else if ( req_part == BLIS_SUBPART1AND0 )
	{
		// A1+A0 (offm,offn) unchanged.
		// A1+A0 is m x (j+b).
		offm_inc = 0;
		offn_inc = 0;
		m_part   = m;
		n_part   = j + b;
	}
	else if ( req_part == BLIS_SUBPART1 )
	{
		// A1 (offm,offn) += (0,j).
		// A1 is m x b.
		offm_inc = 0;
		offn_inc = j;
		m_part   = m;
		n_part   = b;
	}
	else if ( req_part == BLIS_SUBPART1AND2 )
	{
		// A1+A2 (offm,offn) += (0,j).
		// A1+A2 is m x (n-j).
		offm_inc = 0;
		offn_inc = j;
		m_part   = m;
		n_part   = n - j;
	}
	else if ( req_part == BLIS_SUBPART2 ||
	          req_part == subpart2_alias )
	{
		// A2 (offm,offn) += (0,j+b).
		// A2 is m x (n-j-b).
		offm_inc = 0;
		offn_inc = j + b;
		m_part   = m;
		n_part   = n - j - b;
	}


	// Compute the diagonal offset based on the m and n offsets.
	doff_t diagoff_inc = ( doff_t )offm_inc - ( doff_t )offn_inc;


	// Begin by copying the info, elem size, buffer, row stride, and column
	// stride fields of the parent object. Note that this omits copying view
	// information because the new partition will have its own dimensions
	// and offsets.
	bli_obj_init_subpart_from( obj, sub_obj );


	// Modify offsets and dimensions of requested partition based on
	// whether it needs to be transposed.
	if ( bli_obj_has_notrans( obj ) )
	{
		bli_obj_set_dims( m_part, n_part, sub_obj );
		bli_obj_inc_offs( offm_inc, offn_inc, sub_obj );
		bli_obj_inc_diag_offset( diagoff_inc, sub_obj );
	}
	else // if ( bli_obj_has_trans( obj ) )
	{
		bli_obj_set_dims( n_part, m_part, sub_obj );
		bli_obj_inc_offs( offn_inc, offm_inc, sub_obj );
		bli_obj_inc_diag_offset( -diagoff_inc, sub_obj );
	}


	// If the root matrix is not general (ie: has structure defined by the
	// diagonal), and the subpartition does not intersect the root matrix's
	// diagonal, then we might need to modify some of the subpartition's
	// properties, depending on its structure type.
	if ( !bli_obj_root_is_general( sub_obj ) &&
	      bli_obj_is_outside_diag( sub_obj ) )
	{
		// NOTE: This comment may be out-of-date since we now distinguish
		// between uplo properties for the current and root objects...
		// Note that we cannot mark the subpartition object as general/dense
		// here since it makes sense to preserve the existing uplo information
		// a while longer so that the correct kernels are invoked. (Example:
		// incremental packing/computing in gemmt produces subpartitions that
		// appear general/dense, but their uplo fields are needed to be either
		// lower or upper, to determine which macro-kernel gets called in the
		// gemmt_int() back-end.)

		// If the subpartition lies entirely in an "unstored" triangle of the
		// root matrix, then we need to tweak the subpartition. If the root
		// matrix is Hermitian or symmetric, then we reflect the partition to
		// the other side of the diagonal, toggling the transposition bit (and
		// conjugation bit if the root matrix is Hermitian). Or, if the root
		// matrix is triangular, the subpartition should be marked as zero.
		if ( bli_obj_is_unstored_subpart( sub_obj ) )
		{
			if ( bli_obj_root_is_hermitian( sub_obj ) )
			{
				bli_obj_reflect_about_diag( sub_obj );
				bli_obj_toggle_conj( sub_obj );
			}
			else if ( bli_obj_root_is_symmetric( sub_obj ) )
			{
				bli_obj_reflect_about_diag( sub_obj );
			}
			else if ( bli_obj_root_is_triangular( sub_obj ) )
			{
				bli_obj_set_uplo( BLIS_ZEROS, sub_obj );
			}
		}
	}
}


void bli_acquire_mpart_tl2br
     (
             subpart_t req_part,
             dim_t     i,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	bli_acquire_mpart_mndim( BLIS_FWD, req_part, i, b, obj, sub_obj );
}


void bli_acquire_mpart_br2tl
     (
             subpart_t req_part,
             dim_t     j,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	bli_acquire_mpart_mndim( BLIS_BWD, req_part, j, b, obj, sub_obj );
}


void bli_acquire_mpart_mndim
     (
             dir_t     direct,
             subpart_t req_part,
             dim_t     ij,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	dim_t  m;
	dim_t  n;
	dim_t  min_m_n;
	dim_t  m_part   = 0;
	dim_t  n_part   = 0;
	inc_t  offm_inc = 0;
	inc_t  offn_inc = 0;


	// Call a special function for partitioning packed objects. (By only
	// catching those objects packed to panels, we omit cases where the
	// object is packed to row or column storage, as such objects can be
	// partitioned through normally.) Note that the function called below
	// assumes forward partitioning.
	if ( bli_obj_is_panel_packed( obj ) )
		bli_check_error_code( BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_PART );


	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_acquire_mpart_tl2br_check( req_part, ij, b, obj, sub_obj );


	// Query the m and n dimensions of the object (accounting for
	// transposition, if indicated).
	if ( bli_obj_has_notrans( obj ) )
	{
		m = bli_obj_length( obj );
		n = bli_obj_width( obj );
	}
	else // if ( bli_obj_has_trans( obj ) )
	{
		m = bli_obj_width( obj );
		n = bli_obj_length( obj );
	}


	// Foolproofing: do not let b exceed what's left of min(m,n) at
	// row/column offset ij.
	min_m_n = bli_min( m, n );
	if ( b > min_m_n - ij ) b = min_m_n - ij;


	// NOTE: Most of this function implicitly assumes moving forward.
	// When moving backward, we have to relocate ij.
	if ( direct == BLIS_BWD )
	{
		// Modify ij to account for the fact that we are moving backwards.
		ij = min_m_n - ij - b;
	}


	// Compute offset increments and dimensions based on which
	// subpartition is being requested, assuming no transposition.

	// Left column of subpartitions
	if      ( req_part == BLIS_SUBPART00 )
	{
		// A00 (offm,offn) unchanged.
		// A00 is ij x ij.
		offm_inc = 0;
		offn_inc = 0;
		m_part   = ij;
		n_part   = ij;
	}
	else if ( req_part == BLIS_SUBPART10 )
	{
		// A10 (offm,offn) += (ij,0).
		// A10 is b x ij.
		offm_inc = ij;
		offn_inc = 0;
		m_part   = b;
		n_part   = ij;
	}
	else if ( req_part == BLIS_SUBPART20 )
	{
		// A20 (offm,offn) += (ij+b,0).
		// A20 is (m-ij-b) x ij.
		offm_inc = ij + b;
		offn_inc = 0;
		m_part   = m - ij - b;
		n_part   = ij;
	}

	// Middle column of subpartitions.
	else if ( req_part == BLIS_SUBPART01 )
	{
		// A01 (offm,offn) += (0,ij).
		// A01 is ij x b.
		offm_inc = 0;
		offn_inc = ij;
		m_part   = ij;
		n_part   = b;
	}
	else if ( req_part == BLIS_SUBPART11 )
	{
		// A11 (offm,offn) += (ij,ij).
		// A11 is b x b.
		offm_inc = ij;
		offn_inc = ij;
		m_part   = b;
		n_part   = b;
	}
	else if ( req_part == BLIS_SUBPART21 )
	{
		// A21 (offm,offn) += (ij+b,ij).
		// A21 is (m-ij-b) x b.
		offm_inc = ij + b;
		offn_inc = ij;
		m_part   = m - ij - b;
		n_part   = b;
	}

	// Right column of subpartitions.
	else if ( req_part == BLIS_SUBPART02 )
	{
		// A02 (offm,offn) += (0,ij+b).
		// A02 is ij x (n-ij-b).
		offm_inc = 0;
		offn_inc = ij + b;
		m_part   = ij;
		n_part   = n - ij - b;
	}
	else if ( req_part == BLIS_SUBPART12 )
	{
		// A12 (offm,offn) += (ij,ij+b).
		// A12 is b x (n-ij-b).
		offm_inc = ij;
		offn_inc = ij + b;
		m_part   = b;
		n_part   = n - ij - b;
	}
	else // if ( req_part == BLIS_SUBPART22 )
	{
		// A22 (offm,offn) += (ij+b,ij+b).
		// A22 is (m-ij-b) x (n-ij-b).
		offm_inc = ij + b;
		offn_inc = ij + b;
		m_part   = m - ij - b;
		n_part   = n - ij - b;
	}


	// Compute the diagonal offset based on the m and n offsets.
	doff_t diagoff_inc = ( doff_t )offm_inc - ( doff_t )offn_inc;


	// Begin by copying the info, elem size, buffer, row stride, and column
	// stride fields of the parent object. Note that this omits copying view
	// information because the new partition will have its own dimensions
	// and offsets.
	bli_obj_init_subpart_from( obj, sub_obj );


	// Modify offsets and dimensions of requested partition based on
	// whether it needs to be transposed.
	if ( bli_obj_has_notrans( obj ) )
	{
		bli_obj_set_dims( m_part, n_part, sub_obj );
		bli_obj_inc_offs( offm_inc, offn_inc, sub_obj );
		bli_obj_inc_diag_offset( diagoff_inc, sub_obj );
	}
	else // if ( bli_obj_has_trans( obj ) )
	{
		bli_obj_set_dims( n_part, m_part, sub_obj );
		bli_obj_inc_offs( offn_inc, offm_inc, sub_obj );
		bli_obj_inc_diag_offset( -diagoff_inc, sub_obj );
	}

	// If the root matrix is not general (ie: has structure defined by the
	// diagonal), and the subpartition does not intersect the root matrix's
	// diagonal, then set the subpartition structure to "general"; otherwise
	// we let the subpartition inherit the storage structure of its immediate
	// parent.
	if ( !bli_obj_root_is_general( sub_obj ) &&
	     req_part != BLIS_SUBPART00 &&
	     req_part != BLIS_SUBPART11 &&
	     req_part != BLIS_SUBPART22 )
	{
		// FGVZ: Fix me. This needs to be cleaned up. Either non-diagonal
		// intersecting subpartitions should inherit their root object's
		// uplo field, or it should not. Right now, they DO inherit the
		// uplo (because they are not set to BLIS_DENSE when the diagonal
		// does not intersect). But the whole point of being able to query
		// the root object's properties (e.g. uplo field) was so that we
		// COULD mark such subpartitions as dense, to make it easier for
		// certain subproblems on those subpartitions--subproblems that
		// are agnostic to where the subpartition came from.

		// NOTE: This comment may be out-of-date since we now distinguish
		// between uplo properties for the current and root objects...
		// Note that we cannot mark the subpartition object as general/dense
		// here since it makes sense to preserve the existing uplo information
		// a while longer so that the correct kernels are invoked. (Example:
		// incremental packing/computing in gemmt produces subpartitions that
		// appear general/dense, but their uplo fields are needed to be either
		// lower or upper, to determine which macro-kernel gets called in the
		// gemmt_int() back-end.)

		// If the subpartition lies entirely in an "unstored" triangle of the
		// root matrix, then we need to tweak the subpartition. If the root
		// matrix is Hermitian or symmetric, then we reflect the partition to
		// the other side of the diagonal, toggling the transposition bit (and
		// conjugation bit if the root matrix is Hermitian). Or, if the root
		// matrix is triangular, the subpartition should be marked as zero.
		if ( bli_obj_is_unstored_subpart( sub_obj ) )
		{
			if ( bli_obj_root_is_hermitian( sub_obj ) )
			{
				bli_obj_reflect_about_diag( sub_obj );
				bli_obj_toggle_conj( sub_obj );
			}
			else if ( bli_obj_root_is_symmetric( sub_obj ) )
			{
				bli_obj_reflect_about_diag( sub_obj );
			}
			else if ( bli_obj_root_is_triangular( sub_obj ) )
			{
				bli_obj_set_uplo( BLIS_ZEROS, sub_obj );
			}
		}
	}
}


// -- Vector partitioning ------------------------------------------------------


void bli_acquire_vpart_f2b
     (
             subpart_t req_part,
             dim_t     i,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	if ( bli_obj_is_col_vector( obj ) )
		bli_acquire_mpart_mdim( BLIS_FWD, req_part, i, b, obj, sub_obj );
	else // if ( bli_obj_is_row_vector( obj ) )
		bli_acquire_mpart_ndim( BLIS_FWD, req_part, i, b, obj, sub_obj );
}


void bli_acquire_vpart_b2f
     (
             subpart_t req_part,
             dim_t     i,
             dim_t     b,
       const obj_t*    obj,
             obj_t*    sub_obj
     )
{
	if ( bli_obj_is_col_vector( obj ) )
		bli_acquire_mpart_mdim( BLIS_BWD, req_part, i, b, obj, sub_obj );
	else // if ( bli_obj_is_row_vector( obj ) )
		bli_acquire_mpart_ndim( BLIS_BWD, req_part, i, b, obj, sub_obj );
}


// -- Scalar acquisition -------------------------------------------------------


void bli_acquire_mij
     (
             dim_t  i,
             dim_t  j,
       const obj_t* obj,
             obj_t* sub_obj
     )
{
	obj_t tmp_obj;

	bli_acquire_mpart_ndim( BLIS_FWD, BLIS_SUBPART1, j, 1,      obj, &tmp_obj );
	bli_acquire_mpart_mdim( BLIS_FWD, BLIS_SUBPART1, i, 1, &tmp_obj,  sub_obj );
}


void bli_acquire_vi
     (
             dim_t  i,
       const obj_t* obj,
             obj_t* sub_obj
     )
{
	if ( bli_obj_is_col_vector( obj ) )
		bli_acquire_mpart_mdim( BLIS_FWD, BLIS_SUBPART1, i, 1, obj, sub_obj );
	else // if ( bli_obj_is_row_vector( obj ) )
		bli_acquire_mpart_ndim( BLIS_FWD, BLIS_SUBPART1, i, 1, obj, sub_obj );
}

