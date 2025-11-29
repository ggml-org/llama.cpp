/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2021-2025, Southern Methodist University

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

siz_t bli_packm_init
     (
             num_t   dt_p,
       const obj_t*  c,
             obj_t*  p,
       const cntl_t* cntl
     )
{
	bli_init_once();

	// The purpose of packm_init() is to initialize an object P so that
	// a source object A can be packed into P via one of the packm
	// implementations. This initialization precedes the acquisition of a
	// suitable block of memory from the memory allocator (if such a block
	// of memory has not already been allocated previously).

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_packm_init_check( c, p );

	// We begin by copying the fields of A.
	bli_obj_alias_to( c, p );

	// If the object is marked as being filled with zeros, then we can skip
	// the packm operation entirely and alias.
	if ( bli_obj_is_zeros( c ) )
		return false;

	// Extract various fields from the control tree.
	pack_t  schema       = bli_packm_def_cntl_pack_schema( cntl );
	num_t   dt_scalar    = bli_obj_scalar_dt( c );
	dim_t   bmult_m_def  = bli_packm_def_cntl_bmult_m_def( cntl );
	dim_t   bmult_m_pack = bli_packm_def_cntl_bmult_m_pack( cntl );
	dim_t   bmult_n_def  = bli_packm_def_cntl_bmult_n_def( cntl );
	dim_t   bcast_p      = bli_packm_def_cntl_bmult_m_bcast( cntl );

	// Typecast the internal scalar value to the target datatype.
	// Note that if the typecasting is needed, this must happen BEFORE we
	// change the datatype of P to reflect the target_dt.
	if ( dt_scalar != dt_p )
	{
		bli_obj_scalar_cast_to( bli_dt_domain( dt_scalar ) | bli_dt_prec( dt_p ), p );
	}

	// If we are only packing the real part of a complex matrix, use the
	// real datatype for the packed matrix.
	if ( schema == BLIS_PACKED_PANELS_RO )
		dt_p = bli_dt_proj_to_real( dt_p );

	// Update the storage datatype of P to be the target datatype of A.
	bli_obj_set_dt( dt_p, p );
	bli_obj_set_elem_size( bli_dt_size( dt_p ), p );

	// Store the pack schema to the object.
	bli_obj_set_pack_schema( schema, p );

	// Clear the conjugation field from the object since matrix packing
	// in BLIS is deemed to take care of all conjugation necessary.
	bli_obj_set_conj( BLIS_NO_CONJUGATE, p );

	// Since we are packing micropanels, mark P as dense.
	bli_obj_set_uplo( BLIS_DENSE, p );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, p );

	// Compute the dimensions padded by the dimension multiples. These
	// dimensions will be the dimensions of the packed matrices, including
	// zero-padding, and will be used by the macro- and micro-kernels.
	// We compute them by starting with the effective dimensions of A (now
	// in P) and aligning them to the dimension multiples (typically equal
	// to register blocksizes). This does waste a little bit of space for
	// level-2 operations, but that's okay with us.
	dim_t m_p     = bli_obj_length( p );
	dim_t n_p     = bli_obj_width( p );
	dim_t m_p_pad = bli_align_dim_to_mult( m_p, bmult_m_def, true );
	dim_t n_p_pad = bli_align_dim_to_mult( n_p, bmult_n_def, true );

	// Save the padded dimensions into the packed object. It is important
	// to save these dimensions since they represent the actual dimensions
	// of the zero-padded matrix.
	bli_obj_set_padded_dims( m_p_pad, n_p_pad, p );

	// Now we prepare to compute strides, align them, and compute the
	// total number of bytes needed for the packed buffer. Then we use
	// that value to acquire an appropriate block of memory from the
	// memory allocator.

	// Extract the element size for the packed object.
	siz_t elem_size_p = bli_obj_elem_size( p );

	// The panel dimension (for each datatype) should be equal to the
	// default (logical) blocksize multiple in the m dimension.
	dim_t m_panel = bmult_m_def;

	// The "column stride" of a row-micropanel packed object is interpreted
	// as the column stride WITHIN a micropanel. Thus, this is equal to the
	// packing (storage) blocksize multiple, which may be equal to the
	// default (logical) blocksize multiple).
	inc_t cs_p = bmult_m_pack;

	// The "row stride" of a row-micropanel packed object is interpreted
	// as the row stride WITHIN a micropanel. Thus, it is unit unless elemnents
	// are duplicated (broadcast).
	inc_t rs_p = bcast_p;

	// The "panel stride" of a micropanel packed object is interpreted as
	// the distance between the (0,0) element of panel k and the (0,0)
	// element of panel k+1. We use the padded width computed above to
	// allow for zero-padding (if necessary/desired) along the far end
	// of each micropanel (ie: the right edge of the matrix). Zero-padding
	// can also occur along the long edge of the last micropanel if the m
	// dimension of the matrix is not a whole multiple of MR.
	inc_t ps_p = cs_p * n_p_pad;

	// As a general rule, we don't want micropanel strides to be odd. There
	// are very few instances where this can happen, but we've seen it happen
	// more than zero times (such as for certain small problems), and so we
	// check for it here.
	if ( bli_is_odd( ps_p ) ) ps_p += 1;

	// Set the imaginary stride (in units of fundamental elements).
	// This is the number of real elements that must be traversed before
	// reaching the imaginary part of the packed micropanel. NOTE: the
	// imaginary stride is mostly vestigial and left over from the 3m
	// and 4m implementations.
	inc_t is_p = 1;

	// Store the strides and panel dimension in P.
	bli_obj_set_strides( rs_p, cs_p, p );
	bli_obj_set_imag_stride( is_p, p );
	bli_obj_set_panel_dim( m_panel, p );
	bli_obj_set_panel_stride( ps_p, p );
	bli_obj_set_panel_length( m_panel, p );
	bli_obj_set_panel_width( n_p, p );

	// Compute the size of the packed buffer.
	siz_t size_p = ps_p * ( m_p_pad / m_panel ) * elem_size_p;
	return size_p;
}

