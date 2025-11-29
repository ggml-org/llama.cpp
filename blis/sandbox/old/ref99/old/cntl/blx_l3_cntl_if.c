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


void blx_l3_cntl_create_if
     (
       opid_t   family,
       pack_t   schema_a,
       pack_t   schema_b,
       obj_t*   a,
       obj_t*   b,
       obj_t*   c,
       rntm_t*  rntm,
       cntl_t*  cntl_orig,
       cntl_t** cntl_use
     )
{
	// If the control tree pointer is NULL, we construct a default
	// tree as a function of the operation family.
	if ( cntl_orig == NULL )
	{
		*cntl_use = blx_gemm_cntl_create( family, schema_a, schema_b );
	}
	else
	{
		// If the user provided a control tree, create a copy and use it
		// instead (so that threads can use its local tree as a place to
		// cache things like pack mem_t entries).
		*cntl_use = bli_cntl_copy( rntm, cntl_orig );

		// Recursively set the family fields of the newly copied control tree
		// nodes.
		bli_cntl_mark_family( family, *cntl_use );
	}
}

void blx_l3_cntl_free
     (
       rntm_t     rntm,
       cntl_t*    cntl_use,
       thrinfo_t* thread
     )
{
	// If the control tree pointer is NULL, a default tree would have
	// been created, so we now must free it.
	if ( cntl_orig == NULL )
	{
		blx_gemm_cntl_free( rntm, cntl_use, thread );
	}
	else
	{
		// If the user provided a control tree, free the copy of it that
		// was created.
		bli_cntl_free( rntm, cntl_use );
	}
}

