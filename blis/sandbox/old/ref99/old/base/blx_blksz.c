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

dim_t blx_determine_blocksize_f
     (
       dim_t   i,
       dim_t   dim,
       obj_t*  obj,
       bszid_t bszid,
       cntx_t* cntx
     )
{
	num_t    dt;
	blksz_t* bsize;
	dim_t    b_alg, b_max;
	dim_t    b_use;

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize maximum values from the blksz_t object.
	dt    = bli_obj_exec_dt( obj );
	bsize = bli_cntx_get_blksz( bszid, cntx );
	b_alg = bli_blksz_get_def( dt, bsize );
	b_max = bli_blksz_get_max( dt, bsize );

	b_use = blx_determine_blocksize_f_sub( i, dim, b_alg, b_max );

	return b_use;
}

dim_t blx_determine_blocksize_f_sub
     (
       dim_t  i,
       dim_t  dim,
       dim_t  b_alg,
       dim_t  b_max
     )
{
	dim_t b_now;
	dim_t dim_left_now;

	// We assume that this function is being called from an algorithm that
	// is moving "forward" (ie: top to bottom, left to right, top-left
	// to bottom-right).

	// Compute how much of the matrix dimension is left, including the
	// chunk that will correspond to the blocksize we are computing now.
	dim_left_now = dim - i;

	// If the dimension currently remaining is less than the maximum
	// blocksize, use it instead of the default blocksize b_alg.
	// Otherwise, use b_alg.
	if ( dim_left_now <= b_max )
	{
		b_now = dim_left_now;
	}
	else
	{
		b_now = b_alg;
	}

	return b_now;
}

