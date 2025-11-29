/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2021-2024, Southern Methodist University

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

void* bli_packm_alloc
     (
             siz_t      size_needed,
       const cntl_t*    cntl,
             thrinfo_t* thread
     )
{
	// Query the pack buffer type from the control tree node.
	packbuf_t pack_buf_type = bli_packm_def_cntl_pack_buf_type( cntl );

	return bli_packm_alloc_ex
	(
	  size_needed,
	  pack_buf_type,
	  thread
	);
}

void* bli_packm_alloc_ex
     (
       siz_t      size_needed,
       packbuf_t  pack_buf_type,
       thrinfo_t* thread
     )
{
	// Query the address of the mem_t entry within the thrinfo tree node.
	mem_t* mem_p = bli_thrinfo_mem( thread );
	pba_t* pba   = bli_thrinfo_pba( thread );

	mem_t* local_mem_p;
	mem_t  local_mem_s;

	siz_t  mem_size = 0;

	if ( bli_mem_is_alloc( mem_p ) )
		mem_size = bli_mem_size( mem_p );

	if ( mem_size < size_needed )
	{
		if ( bli_thrinfo_am_chief( thread ) )
		{
			// The chief thread releases the existing block associated with
			// the mem_t entry in the thrinfo tree, and then re-acquires a
			// new block, saving the associated mem_t entry to local_mem_s.
			if ( bli_mem_is_alloc( mem_p ) )
			{
				bli_pba_release
				(
				  pba,
				  mem_p
				);
			}

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
		// this thread's thrinfo tree node.
		*mem_p = *local_mem_p;

		// Barrier so that the master thread doesn't return from the function
		// before we are done reading.
		bli_thrinfo_barrier( thread );
	}

	return bli_mem_buffer( mem_p );
}

