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

cntl_t* blx_packm_cntl_create_node
     (
	   rntm_t*   rntm,
       void_fp   var_func,
       void_fp   packm_var_func,
       bszid_t   bmid_m,
       bszid_t   bmid_n,
       bool_t    does_invert_diag,
       bool_t    rev_iter_if_upper,
       bool_t    rev_iter_if_lower,
       pack_t    pack_schema,
       packbuf_t pack_buf_type,
       cntl_t*   sub_node
     )
{
	cntl_t*         cntl;
	packm_params_t* params;

	// Allocate a packm_params_t struct.
	params = bli_malloc_intl( sizeof( packm_params_t ) );

	// Initialize the packm_params_t struct.
	params->size              = sizeof( packm_params_t );
	params->var_func          = packm_var_func;
	params->bmid_m            = bmid_m;
	params->bmid_n            = bmid_n;
	params->does_invert_diag  = does_invert_diag;
	params->rev_iter_if_upper = rev_iter_if_upper;
	params->rev_iter_if_lower = rev_iter_if_lower;
	params->pack_schema       = pack_schema;
	params->pack_buf_type     = pack_buf_type;

	// It's important that we set the bszid field to BLIS_NO_PART to indicate
	// that no blocksize partitioning is performed. bli_cntl_free() will rely
	// on this information to know how to step through the thrinfo_t tree in
	// sync with the cntl_t tree.
	cntl = bli_cntl_create_node
	(
	  BLIS_NOID,
	  BLIS_NO_PART,
	  var_func,
	  params,
	  sub_node
	);

	return cntl;
}

