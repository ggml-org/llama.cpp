/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin
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

//
// Variant 1 provides basic support for packing by calling packm_cxk().
//

void bls_packm_var1
     (
             num_t      dt,
             conj_t     conjc,
             dim_t      m,
             dim_t      n,
             dim_t      m_max,
             dim_t      n_max,
       const void*      kappa,
       const void*      c, inc_t rs_c, inc_t cs_c,
             void*      p, inc_t rs_p, inc_t cs_p,
                           dim_t pd_p, inc_t ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
	( void )m_max;
	( void )rs_p;

	const char* c_cast  = c;
	      char* p_cast  = p;

	dim_t iter_dim      = m;
	dim_t panel_len     = n;
	dim_t panel_len_max = n_max;
	dim_t panel_dim_max = pd_p;
	inc_t incc          = rs_c;
	inc_t ldc           = cs_c;
	inc_t ldp           = cs_p;
	dim_t dt_size       = bli_dt_size( dt );

	packm_cxk_ker_ft f  = bli_cntx_get_ukr2_dt( dt, dt, BLIS_PACKM_KER, cntx );

	// Compute the total number of iterations we'll need.
	dim_t n_iter  = ( iter_dim + panel_dim_max - 1) / panel_dim_max;
	char* p_begin = p_cast;

	// Query the number of threads and thread ids from the current thread's
	// packm thrinfo_t node.
	const dim_t nt  = bli_thrinfo_num_threads( thread );
	const dim_t tid = bli_thrinfo_thread_id( thread );

	// Suppress warnings in case tid isn't used (ie: as in slab partitioning).
	( void )nt;
	( void )tid;

	// Determine the thread range and increment using the current thread's
	// packm thrinfo_t node. NOTE: The definition of bli_thread_range_slrr()
	// will depend on whether slab or round-robin partitioning was requested
	// at configure-time.
	dim_t it_start, it_end, it_inc;
	bli_thread_range_slrr( tid, nt, n_iter, 1, FALSE, &it_start, &it_end, &it_inc );

	// Iterate over every logical micropanel in the source matrix.
	for ( dim_t ic  = 0, it  = 0; it < n_iter; ic += panel_dim_max, it += 1 )
	{
		dim_t panel_dim = bli_min( panel_dim_max, iter_dim - ic );

		const char* c_begin = c_cast + ic*incc*dt_size;

		const char* c_use = c_begin;
		      char* p_use = p_begin;

		// The definition of bli_is_my_iter() will depend on whether slab
		// or round-robin partitioning was requested at configure-time. (The
		// default is slab.)
		if ( bli_is_my_iter( it, it_start, it_end, tid, nt ) )
		{
			f
			(
			  conjc,
			  BLIS_PACKED_PANELS,
			  panel_dim,
			  panel_dim_max,
			  1, // TODO: this shouldn't be hard-coded.
			  panel_len,
			  panel_len_max,
			  kappa,
			  c_use, incc, ldc,
			  p_use,       ldp,
			  NULL,
			  cntx
			);

			// The packing microkernel f is equivalent to (where ctype and ch represent
			// the data type and type character, e.g. `float` and `s`):
			//
			//  ctype  kappa_cast = *( ctype* )kappa;
			//  ctype* c_cast     = ( ctype* )c_use;
			//  ctype* p_cast     = ( ctype* )p_use;
			//
			//  // Perform the packing, taking conjc into account.
			//  if ( bli_is_conj( conjc ) )
			//  {
			//  	for ( dim_t l = 0; l < panel_len; ++l )
			//  	{
			//  		for ( dim_t i = 0; i < panel_dim; ++i )
			//  		{
			//  			ctype* cli = c_cast + (l  )*ldc + (i  )*incc;
			//  			ctype* pli = p_cast + (l  )*ldp + (i  )*1;
			//
			//  			bli_taxpyjs( ch,ch,ch,ch, kappa_cast, *cli, *pli );
			//  		}
			//  	}
			//  }
			//  else
			//  {
			//  	for ( dim_t l = 0; l < panel_len; ++l )
			//  	{
			//  		for ( dim_t i = 0; i < panel_dim; ++i )
			//  		{
			//  			ctype* cli = c_cast + (l  )*ldc + (i  )*incc;
			//  			ctype* pli = p_cast + (l  )*ldp + (i  )*1;
			//
			//  			bli_taxpys( ch,ch,ch,ch, kappa_cast, *cli, *pli );
			//  		}
			//  	}
			//  }
			//
			//  // If panel_dim < panel_dim_max and/or panel_len < panel_len_max,
			//  // then we zero those unused rows/columns.
			//  PASTEMAC(ch,set0s_edge)
			//  (
			//    panel_dim, panel_dim_max,
			//    panel_len, panel_len_max,
			//    p_cast, ldp
			//  );
		}

		p_begin += ps_p*dt_size;
	}
}

