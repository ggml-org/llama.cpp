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

// -- Macros to help concisely instantiate bli_func_init() ---------------------

#define gen_func_init_ro( func_p, opname ) \
do { \
	bli_func_init( func_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                       NULL,               NULL ); \
} while (0)

#define gen_func_init_co( func_p, opname ) \
do { \
	bli_func_init( func_p, NULL,               NULL, \
	                       PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
} while (0)

#define gen_func_init( func_p, opname ) \
do { \
	bli_func_init( func_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                       PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
} while (0)

// -----------------------------------------------------------------------------

void PASTEMAC(plugin_init_@plugin_name@,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)
     (
       plugin_@plugin_name@_params
     )
{
	cntx_t* cntx = ( cntx_t* )bli_gks_lookup_id( PASTECH(BLIS_ARCH,BLIS_CNAME_UPPER_INFIX) );
	( void )cntx;

    // ------------------------------------------------------------------------>
	// -- Example Initialization ---------------------------------------------->
	// ------------------------------------------------------------------------>

	blksz_t blkszs[ MY_NUM_BLOCK_SIZES ];
	kerid_t bmults[ MY_NUM_BLOCK_SIZES ];
	func_t  funcs[ MY_NUM_KERNELS ];
	mbool_t mbools[ MY_NUM_KERNEL_PREFS ];

	// -- Set blocksizes -------------------------------------------------------
    //                                             s     d     c     z
	bli_blksz_init_easy( &blkszs[ MY_BLKSZ_1 ],  256,  128,  128,   64 );
	bli_blksz_init_easy( &blkszs[ MY_BLKSZ_2 ],  256,  256,  256,  256 );
	bmults[ MY_BLKSZ_1 ] = bszids[ MY_BLKSZ_1 ];
	bmults[ MY_BLKSZ_2 ] = bszids[ MY_BLKSZ_2 ];

	// -- Set micro-kernels ----------------------------------------------------

	gen_func_init   ( &funcs[ MY_KERNEL_1 ], PASTECH(my_kernel_1,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX) );
	gen_func_init_co( &funcs[ MY_KERNEL_2 ], PASTECH(my_kernel_2,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX) );

	// -- Set preferences ------------------------------------------------------
	//                                        s      d      c      z
	bli_mbool_init( &mbools[ MY_PREF_1 ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ MY_PREF_2 ], FALSE, FALSE, FALSE, FALSE );

	// -- Put block sizes, kernels, and preferences into the context -----------

	for ( dim_t i = 0; i < MY_NUM_BLOCK_SIZES; i++ )
		bli_cntx_set_blksz( bszids[ i ], &blkszs[ i ], bmults[ i ], cntx );

	for ( dim_t i = 0; i < MY_NUM_KERNELS; i++ )
		bli_cntx_set_ukr( kerids[ i ], &funcs[ i ], cntx );

	for ( dim_t i = 0; i < MY_NUM_KERNEL_PREFS; i++ )
		bli_cntx_set_ukr_pref( prefids[ i ], &mbools[ i ], cntx );

	// <------------------------------------------------------------------------
	// <------------------------------------------------------------------------
	// <------------------------------------------------------------------------
}

