/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

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

void bls_packm_int
     (
             num_t      dt,
             conj_t     conj,
             dim_t      m_alloc,
             dim_t      k_alloc,
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
	/* Set the pack buffer type so that we are obtaining memory blocks from
	   the pool dedicated to blocks of A. */
	const packbuf_t pack_buf_type = BLIS_BUFFER_FOR_A_BLOCK;

	/* NOTE: This "rounding up" of the last upanel is absolutely necessary since
	   we NEED that last micropanel to have the same ldim (cs_p) as the other
	   micropanels. Why? Because the microkernel assumes that the register (MR,
	   NR) AND storage (PACKMR, PACKNR) blocksizes do not change. */
	const dim_t m_pack = ( ( m_alloc + mr - 1) / mr ) * mr;
	const dim_t k_pack = k_alloc;

	/* Barrier to make sure all threads are caught up and ready to begin the
	   packm stage. */
	bli_thrinfo_barrier( thread );

	/* Compute the size of the memory block eneded. */
	siz_t size_needed = bli_dt_size( dt ) * m_pack * k_pack;

	/* NOTE: This "rounding up" of the last upanel is absolutely necessary since
	   we NEED that last micropanel to have the same ldim (cs_p) as the other
	   micropanels. Why? Because the microkernel assumes that the register (MR,
	   NR) AND storage (PACKMR, PACKNR) blocksizes do not change. */
	dim_t m_max = ( m / mr + ( m % mr ? 1 : 0 ) ) * mr;
	dim_t k_max = k;

	// Determine the dimensions and strides for the packed matrix.
	*rs_p = 1;
	*cs_p = mr;

	dim_t pd_p = mr;
	*ps_p = mr * k;

	/* Set the buffer address provided by the caller to point to the memory
	   associated with the mem_t entry acquired from the memory pool. */
	*p = bli_packm_alloc_ex
	(
	  size_needed,
	  pack_buf_type,
	  thread
	);

	bls_packm_var1
	(
	  dt,
	  conj,
	  m,
	  k,
	  m_max,
	  k_max,
	  kappa,
	  a,  rs_a,  cs_a,
	  *p, *rs_p, *cs_p,
	       pd_p, *ps_p,
	  cntx,
	  thread
	);
}


//
// Define BLAS-like interfaces to the variant chooser.
//

void bls_packm_a
     (
             num_t      dt,
             conj_t     conj,
             dim_t      m_alloc,
             dim_t      k_alloc,
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
	bls_packm_int
	(
	  dt,
	  conj,
	  m_alloc,
	  k_alloc,
	  m,
	  k,
	  mr,
	  kappa,
	  a, rs_a, cs_a,
	  p, rs_p, cs_p,
	     ps_p,
	  cntx,
	  thread
	);

	/* Barrier so that packing is done before computation. */
	bli_thrinfo_barrier( thread );
}

void bls_packm_b
     (
             num_t      dt,
             conj_t     conj,
             dim_t      k_alloc,
             dim_t      n_alloc,
             dim_t      k,
             dim_t      n,
             dim_t      nr,
       const void*      kappa,
       const void*      b, inc_t  rs_b, inc_t  cs_b,
             void**     p, inc_t* rs_p, inc_t* cs_p,
                           inc_t* ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	// Implicitly transpose B for packing.
	bls_packm_int
	(
	  dt,
	  conj,
	  n_alloc,
	  k_alloc,
	  n,
	  k,
	  nr,
	  kappa,
	  b, cs_b, rs_b,
	  p, cs_p, rs_p,
	     ps_p,
	  cntx,
	  thread
	);

	/* Barrier so that packing is done before computation. */
	bli_thrinfo_barrier( thread );
}
