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
#include "blix.h"

void blx_l3_packm
     (
       obj_t*  x,
       obj_t*  x_pack,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	pba_t* pba;
	packbuf_t pack_buf_type;
	mem_t*    cntl_mem_p;
	siz_t     size_needed;

	// FGVZ: Not sure why we need this barrier, but we do.
	bli_thrinfo_barrier( thread );

	// Every thread initializes x_pack and determines the size of memory
	// block needed (which gets embedded into the otherwise "blank" mem_t
	// entry in the control tree node).
	size_needed
	=
	bli_packm_init
	(
	  x,
	  x_pack,
	  cntx,
	  cntl
	);

	// If zero was returned, no memory needs to be allocated and so we can
	// return early.
	if ( size_needed == 0 ) return;

	// Query the memory broker from the context.
	pba = bli_cntx_get_pba( cntx );

	// Query the pack buffer type from the control tree node.
	pack_buf_type = bli_cntl_packm_params_pack_buf_type( cntl );

	// Query the address of the mem_t entry within the control tree node.
	cntl_mem_p = bli_cntl_pack_mem( cntl );

	// Check the mem_t field in the control tree. If it is unallocated, then
	// we need to acquire a block from the memory broker and broadcast it to
	// all threads in the chief's thread group.
	if ( bli_mem_is_unalloc( cntl_mem_p ) )
	{
		mem_t* local_mem_p;
		mem_t  local_mem_s;

		if ( bli_thread_am_ochief( thread ) )
		{
			// The chief thread acquires a block from the memory broker
			// and saves the associated mem_t entry to local_mem_s.
			bli_pba_acquire_m
			(
			  pba,
			  size_needed,
			  pack_buf_type,
			  &local_mem_s
			);
		}

		// Broadcast the address of the chief thread's local mem_t entry to
		// all threads.
		local_mem_p = bli_thrinfo_broadcast( thread, &local_mem_s );

		// Save the contents of the chief thread's local mem_t entry to the
		// mem_t field in this thread's control tree node.
		*cntl_mem_p = *local_mem_p;
	}
	else // ( bli_mem_is_alloc( cntl_mem_p ) )
	{
		mem_t* local_mem_p;
		mem_t  local_mem_s;

		// If the mem_t entry in the control tree does NOT contain a NULL
		// buffer, then a block has already been acquired from the memory
		// broker and cached in the control tree.

		// As a sanity check, we should make sure that the mem_t object isn't
		// associated with a block that is too small compared to the size of
		// the packed matrix buffer that is needed, according to the return
		// value from packm_init().
		siz_t cntl_mem_size = bli_mem_size( cntl_mem_p );

		if ( cntl_mem_size < size_needed )
		{
			if ( bli_thread_am_ochief( thread ) )
			{
				// The chief thread releases the existing block associated with
				// the mem_t entry in the control tree, and then re-acquires a
				// new block, saving the associated mem_t entry to local_mem_s.
				bli_pba_release( cntl_mem_p );
				bli_pba_acquire_m
				(
				  pba,
				  size_needed,
				  pack_buf_type,
				  &local_mem_s
				);
			}

			// Broadcast the address of the chief thread's local mem_t entry to
			// all threads.
			local_mem_p = bli_thrinfo_broadcast( thread, &local_mem_s );

			// Save the chief thread's local mem_t entry to the mem_t field in
			// this thread's control tree node.
			*cntl_mem_p = *local_mem_p;
		}
		else
		{
			// If the mem_t entry is already allocated and sufficiently large,
			// then we use it as-is. No action is needed, because all threads
			// will already have the cached values in their local control
			// trees' mem_t entries, currently pointed to by cntl_mem_p.

			bli_thrinfo_barrier( thread );
		}
	}


	// Update the buffer address in x_pack to point to the buffer associated
	// with the mem_t entry acquired from the memory broker (now cached in
	// the control tree node).
	void* buf = bli_mem_buffer( cntl_mem_p );
    bli_obj_set_buffer( buf, x_pack );


	// Pack the contents of object x to object x_pack.
	bli_packm_int
	(
	  x,
	  x_pack,
	  cntx,
	  cntl,
	  thread
	);

	// Barrier so that packing is done before computation.
	bli_thrinfo_barrier( thread );
}

