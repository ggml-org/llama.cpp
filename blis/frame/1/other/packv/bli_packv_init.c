/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP

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

void bli_packv_init
     (
       obj_t*   a,
       obj_t*   p,
       cntx_t*  cntx,
       packv_t* cntl
     )
{
	// The purpose of packm_init() is to initialize an object P so that

	// a source object A can be packed into P via one of the packv
	// implementations. This initialization includes acquiring a suitable
	// block of memory from the memory allocator, if such a block of memory
	// has not already been allocated previously.

	pack_t   pack_schema;
	bszid_t  bmult_id;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_packv_check( a, p, cntx );

	// First check if we are to skip this operation because the control tree
	// is NULL, and if so, simply alias the object to its packed counterpart.
	if ( bli_cntl_is_noop( cntl ) )
	{
		bli_obj_alias_to( a, p );
		return;
	}

	// At this point, we can be assured that cntl is not NULL. Let us now
	// check to see if the object has already been packed to the desired
	// schema (as encoded in the control tree). If so, we can alias and
	// return, as above.
	// Note that in most cases, bli_obj_pack_schema() will return
	// BLIS_NOT_PACKED and thus packing will be called for (but in some
	// cases packing has already taken place). Also, not all combinations
	// of current pack status and desired pack schema are valid.
	if ( bli_obj_pack_schema( a ) == cntl_pack_schema( cntl ) )
	{
		bli_obj_alias_to( a, p );
		return;
	}

	// Now, if we are not skipping the pack operation, then the only question
	// left is whether we are to typecast vector a before packing.
	if ( bli_obj_dt( a ) != bli_obj_target_dt( a ) )
		bli_abort();

	// Extract various fields from the control tree and pass them in
	// explicitly into _init_pack(). This allows external code generators
	// the option of bypassing usage of control trees altogether.
	pack_schema = cntl_pack_schema( cntl );
	bmult_id    = cntl_bmid( cntl );

	// Initialize object p for the final packed vector.
	bli_packv_init_pack
	(
	  pack_schema,
	  bmult_id,
	  &a,
	  p,
	  cntx
	);

	// Now p is ready to be packed.
}


siz_t bli_packv_init_pack
     (
       pack_t  schema,
       bszid_t bmult_id,
       obj_t*  a,
       obj_t*  p,
       cntx_t* cntx
     )
{
	num_t     dt     = bli_obj_dt( a );
	dim_t     dim_a  = bli_obj_vector_dim( a );
	dim_t     bmult  = bli_cntx_get_blksz_def_dt( dt, bmult_id, cntx );

	pba_t*    pba    = bli_cntx_pba( cntx );

#if 0
	mem_t*    mem_p;
#endif
	dim_t     m_p_pad;
	siz_t     size_p;
	inc_t     rs_p, cs_p;
	void*     buf;


	// We begin by copying the basic fields of c.
	bli_obj_alias_to( a, p );

	// Update the dimensions.
	bli_obj_set_dims( dim_a, 1, p );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, p );

	// Set the pack schema in the p object to the value in the control tree
	// node.
	bli_obj_set_pack_schema( schema, p );

	// Compute the dimensions padded by the dimension multiples.
	m_p_pad = bli_align_dim_to_mult( bli_obj_vector_dim( p ), bmult );

	// Compute the size of the packed buffer.
	size_p = m_p_pad * 1 * bli_obj_elem_size( p );

#if 0
	// Extract the address of the mem_t object within p that will track
	// properties of the packed buffer.
	mem_p = bli_obj_pack_mem( *p );

	if ( bli_mem_is_unalloc( mem_p ) )
	{
		// If the mem_t object of p has not yet been allocated, then acquire
		// a memory block suitable for a vector.
		bli_pba_acquire_v( pba, size_p, mem_p );
	}
	else
	{
 		// If the mem_t object has already been allocated, then release and
		// re-acquire the memory so there is sufficient space.
		if ( bli_mem_size( mem_p ) < size_p )
		{
			bli_pba_release( mem_p );

			bli_pba_acquire_v( pba, size_p, mem_p );
		}
	}

	// Grab the buffer address from the mem_t object and copy it to the
	// main object buffer field. (Sometimes this buffer address will be
	// copied when the value is already up-to-date, because it persists
	// in the main object buffer field across loop iterations.)
	buf = bli_mem_buffer( mem_p );
	bli_obj_set_buffer( buf, p );
#endif

	// Save the padded (packed) dimensions into the packed object.
	bli_obj_set_padded_dims( m_p_pad, 1, p );

	// Set the row and column strides of p based on the pack schema.
	if ( schema == BLIS_PACKED_VECTOR )
	{
		// Set the strides to reflect a column-stored vector. Note that the
		// column stride may never be used, and is only useful to determine
		// how much space beyond the vector would need to be zero-padded, if
		// zero-padding was needed.
		rs_p = 1;
		cs_p = bli_obj_padded_length( p );

		bli_obj_set_strides( rs_p, cs_p, p );
	}

	return size_p;
}

#if 0
void bli_packv_release
     (
       obj_t*   p,
       packv_t* cntl
     )
{
	if ( !bli_cntl_is_noop( cntl ) )
		bli_obj_release_pack( p );
}
#endif
