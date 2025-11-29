/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.
   Copyright (C) 2024, Southern Methodist University

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

void bli_packm_sup_finalize_mem
     (
       bool       did_pack,
       thrinfo_t* thread
     )
{
	// Inspect whether we previously packed matrix A.
	if ( did_pack == FALSE )
	{
		// If we didn't pack matrix A, there's nothing to be done.
	}
	else // if ( did_pack == TRUE )
	{
		mem_t* mem = bli_thrinfo_mem( thread );
		pba_t* pba = bli_thrinfo_pba( thread );

		if ( thread != NULL )
		if ( bli_thrinfo_am_chief( thread ) )
		{
			// Check the mem_t entry provided by the caller. Only proceed if it
			// is allocated, which it should be.
			if ( bli_mem_is_alloc( mem ) )
			{
				bli_pba_release
				(
				  pba,
				  mem
				);
			}
		}
	}
}

typedef void (*packm_sup_var1_fp)
     (
       trans_t    transc,
       pack_t     schema,
       dim_t      m,
       dim_t      n,
       dim_t      m_max,
       dim_t      n_max,
       void*      kappa,
       void*      c, inc_t rs_c, inc_t cs_c,
       void*      p, inc_t rs_p, inc_t cs_p,
                           dim_t pd_p, inc_t ps_p,
       cntx_t*    cntx,
       thrinfo_t* thread
     );

typedef void (*packm_sup_var2_fp)
     (
       trans_t    transc,
       pack_t     schema,
       dim_t      m,
       dim_t      n,
       void*      kappa,
       void*      c, inc_t rs_c, inc_t cs_c,
       void*      p, inc_t rs_p, inc_t cs_p,
       cntx_t*    cntx,
       thrinfo_t* thread
     );

static packm_sup_var1_fp GENARRAY(packm_sup_var1,packm_sup_var1);
static packm_sup_var2_fp GENARRAY(packm_sup_var2,packm_sup_var2);

//
// Define BLAS-like interfaces to the variant chooser.
//

void bli_packm_sup
     (
             bool       will_pack,
             packbuf_t  pack_buf_type,
             stor3_t    stor_id,
             num_t      dt,
             dim_t      m,
             dim_t      k,
             dim_t      mr,
       const void*      kappa,
       const void*      a, inc_t  rs_a, inc_t  cs_a,
             void**     p, inc_t* rs_p, inc_t* cs_p,
                           inc_t* ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	if ( will_pack == FALSE )
	{
		// Set the parameters for use with no packing of A (ie: using the
		// source matrix A directly).

		// Use the strides of the source matrix as the final values.
		*rs_p = rs_a;
		*cs_p = cs_a;
		*ps_p = mr * rs_a;

		// Since we won't be packing, simply update the buffer address provided
		// by the caller to point to source matrix.
		*p = ( void* )a;

		return;
	}

	// Barrier so that computation is done before packing.
	bli_thrinfo_barrier( thread );

	// NOTE: This is "rounding up" of the last upanel is actually optional
	// for the rrc/crc cases, but absolutely necessary for the other cases
	// since we NEED that last micropanel to have the same ldim (cs_p) as
	// the other micropanels. Why? So that millikernels can use the same
	// upanel ldim for all iterations of the ir loop.
	dim_t  m_max = ( m / mr + ( m % mr ? 1 : 0 ) ) * mr;
	dim_t  k_max = k;

	dim_t  pd_p = mr;
	*ps_p = mr * k;

	pack_t schema;

	// Determine the dimensions and strides for the packed matrix A.
	if ( stor_id == BLIS_RRC ||
		 stor_id == BLIS_CRC )
	{
		// stor3_t id values _RRC and _CRC: pack A to plain row storage.
		*rs_p = k;
		*cs_p = 1;

		// Set the schema to "row packed" to indicate packing to plain
		// row storage.
		schema = BLIS_PACKED_MATRIX;
	}
	else
	{
		// All other stor3_t ids: pack A to column-stored row-panels.
		*rs_p = 1;
		*cs_p = mr;

		// Set the schema to "packed row panels" to indicate packing to
		// conventional column-stored row panels.
		schema = BLIS_PACKED_PANELS;
	}

	// NOTE: This "rounding up" of the last upanel is actually optional
	// for the rrc/crc cases, but absolutely necessary for the other cases
	// since we NEED that last micropanel to have the same ldim (cs_p) as
	// the other micropanels. Why? So that millikernels can use the same
	// upanel ldim for all iterations of the ir loop.
	const dim_t m_pack = ( m / mr + ( m % mr ? 1 : 0 ) ) * mr;
	const dim_t k_pack = k;

	// Compute the size of the memory block eneded.
	siz_t size_needed = bli_dt_size( dt ) * m_pack * k_pack;

	// Set the buffer address provided by the caller to point to the
	// memory associated with the mem_t entry acquired from the pba.
	*p = bli_packm_alloc_ex( size_needed, pack_buf_type, thread );

	if ( schema == BLIS_PACKED_MATRIX )
	{
		// printf( "blis_ packm_sup_a: packing A to rows.\n" );

		// For plain packing by rows, use var2.
		packm_sup_var2[ dt ]
		(
		  BLIS_NO_TRANSPOSE,
		  schema,
		  m,
		  k,
		  ( void* )kappa,
		  ( void* )a,  rs_a,  cs_a,
		          *p, *rs_p, *cs_p,
		  ( cntx_t* )cntx,
		  thread
		);
	}
	else // if ( schema == BLIS_PACKED_PANELS )
	{
		// printf( "blis_ packm_sup_a: packing A to row panels.\n" );

		// For packing to column-stored row panels, use var1.
		packm_sup_var1[ dt ]
		(
		  BLIS_NO_TRANSPOSE,
		  schema,
		  m,
		  k,
		  m_max,
		  k_max,
		  ( void* )kappa,
		  ( void* )a,  rs_a,  cs_a,
		          *p, *rs_p, *cs_p,
		               pd_p, *ps_p,
		  ( cntx_t* )cntx,
		  thread
		);
	}

	// Barrier so that packing is done before computation.
	bli_thrinfo_barrier( thread );
}

