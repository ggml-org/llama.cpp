/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Southern Methodist University

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

#include @PLUGIN_HEADER@

void PASTEMAC(plugin_init_@plugin_name@,BLIS_CNAME_INFIX)
     (
       plugin_@plugin_name@_params
     )
{
	cntx_t* cntx = ( cntx_t* )bli_gks_lookup_id( PASTECH(BLIS_ARCH,BLIS_CNAME_UPPER_INFIX) );
	( void )cntx;

	PASTEMAC(plugin_init_@plugin_name@,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)
    (
      plugin_@plugin_name@_params_only
    );

    // ------------------------------------------------------------------------>
	// -- Example Initialization ---------------------------------------------->
	// ------------------------------------------------------------------------>

	// Update the context with optimized native micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  kerids[ MY_KERNEL_1 ], BLIS_DOUBLE, bli_dmy_kernel_1_zen3,

	  BLIS_VA_END
	);

	// Update the context with preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  prefids[ MY_PREF_1 ], BLIS_DOUBLE, TRUE,
	  prefids[ MY_PREF_2 ], BLIS_DOUBLE, TRUE,

	  BLIS_VA_END
	);

	blksz_t blkszs[ MY_NUM_BLOCK_SIZES ];
	bszid_t bmults[ MY_NUM_BLOCK_SIZES ];

	// Update block sizes
	//                                             s     d     c     z
	bli_blksz_init_easy( &blkszs[ MY_BLKSZ_1 ],  320,  240,  182,   96 );
	bmults[ MY_BLKSZ_1 ] = bszids[ MY_BLKSZ_1 ];

	bli_cntx_set_blkszs
	(
	  cntx,

	  bszids[ MY_BLKSZ_1 ], &blkszs[ MY_BLKSZ_1 ], bmults[ MY_BLKSZ_1 ],

	  BLIS_VA_END
	);

	// <------------------------------------------------------------------------
	// <------------------------------------------------------------------------
	// <------------------------------------------------------------------------
}

