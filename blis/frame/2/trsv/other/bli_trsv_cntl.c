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

extern scalv_t*   scalv_cntl;
extern packm_t*   packm_cntl;
extern packv_t*   packv_cntl;
extern unpackv_t* unpackv_cntl;

extern gemv_t*    gemv_cntl_rp_bs_dot;
extern gemv_t*    gemv_cntl_rp_bs_axpy;
extern gemv_t*    gemv_cntl_cp_bs_dot;
extern gemv_t*    gemv_cntl_cp_bs_axpy;

trsv_t*           trsv_cntl_bs_ke_nrow_tcol = NULL;
trsv_t*           trsv_cntl_bs_ke_ncol_trow = NULL;
trsv_t*           trsv_cntl_ge_nrow_tcol = NULL;
trsv_t*           trsv_cntl_ge_ncol_trow = NULL;


void bli_trsv_cntl_init()
{
	// Create control trees for the lowest-level kernels. These trees induce
	// operations on (presumably) relatively small block-subvector problems.
	trsv_cntl_bs_ke_nrow_tcol
	=
	bli_trsv_cntl_obj_create( BLIS_UNB_FUSED,
	                          BLIS_VARIANT1,
	                          0,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL,
	                          NULL );
	trsv_cntl_bs_ke_ncol_trow
	=
	bli_trsv_cntl_obj_create( BLIS_UNB_FUSED,
	                          BLIS_VARIANT2,
	                          0,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL,
	                          NULL );

	// Create control trees for generally large problems. Here we choose a
	// variant that prioritizes keeping a subvector of x in cache.
	trsv_cntl_ge_nrow_tcol
    =
	bli_trsv_cntl_obj_create( BLIS_BLOCKED,
                              BLIS_VARIANT1,        // use var1 to maximize x1 usage
                              BLIS_M2,
                              scalv_cntl,           // scale x up-front
                              packm_cntl,           // pack A11 (if needed)
                              packv_cntl,           // pack x1 (if needed)
                              gemv_cntl_rp_bs_dot,  // gemv_rp needed by var1
                              NULL,                 // gemv_cp not needed by var1
                              trsv_cntl_bs_ke_nrow_tcol,
                              unpackv_cntl );       // unpack x1 (if needed)
	trsv_cntl_ge_ncol_trow
    =
	bli_trsv_cntl_obj_create( BLIS_BLOCKED,
                              BLIS_VARIANT1,        // use var1 to maximize x1 usage
                              BLIS_M2,
                              scalv_cntl,           // scale x up-front
                              packm_cntl,           // pack A11 (if needed)
                              packv_cntl,           // pack x1 (if needed)
                              gemv_cntl_rp_bs_axpy, // gemv_rp needed by var1
                              NULL,                 // gemv_cp not needed by var1
                              trsv_cntl_bs_ke_ncol_trow,
                              unpackv_cntl );       // unpack x1 (if needed)
}

void bli_trsv_cntl_finalize()
{
	bli_cntl_free_node( trsv_cntl_bs_ke_nrow_tcol );
	bli_cntl_free_node( trsv_cntl_bs_ke_ncol_trow );
	bli_cntl_free_node( trsv_cntl_ge_nrow_tcol );
	bli_cntl_free_node( trsv_cntl_ge_ncol_trow );
}


trsv_t* bli_trsv_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  bszid_t    bszid,
                                  scalv_t*   sub_scalv,
                                  packm_t*   sub_packm_a11,
                                  packv_t*   sub_packv_x1,
                                  gemv_t*    sub_gemv_rp,
                                  gemv_t*    sub_gemv_cp,
                                  trsv_t*    sub_trsv,
                                  unpackv_t* sub_unpackv_x1 )
{
	trsv_t* cntl;

	cntl = ( trsv_t* ) bli_malloc_intl( sizeof(trsv_t) );

	cntl->impl_type      = impl_type;
	cntl->var_num        = var_num;
	cntl->bszid          = bszid;
	cntl->sub_scalv      = sub_scalv;
	cntl->sub_packm_a11  = sub_packm_a11;
	cntl->sub_packv_x1   = sub_packv_x1;
	cntl->sub_gemv_rp    = sub_gemv_rp;
	cntl->sub_gemv_cp    = sub_gemv_cp;
	cntl->sub_trsv       = sub_trsv;
	cntl->sub_unpackv_x1 = sub_unpackv_x1;

	return cntl;
}

void bli_trsv_cntl_obj_init( trsv_t*    cntl,
                             impl_t     impl_type,
                             varnum_t   var_num,
                             bszid_t    bszid,
                             scalv_t*   sub_scalv,
                             packm_t*   sub_packm_a11,
                             packv_t*   sub_packv_x1,
                             gemv_t*    sub_gemv_rp,
                             gemv_t*    sub_gemv_cp,
                             trsv_t*    sub_trsv,
                             unpackv_t* sub_unpackv_x1 )
{
	cntl->impl_type      = impl_type;
	cntl->var_num        = var_num;
	cntl->bszid          = bszid;
	cntl->sub_scalv      = sub_scalv;
	cntl->sub_packm_a11  = sub_packm_a11;
	cntl->sub_packv_x1   = sub_packv_x1;
	cntl->sub_gemv_rp    = sub_gemv_rp;
	cntl->sub_gemv_cp    = sub_gemv_cp;
	cntl->sub_trsv       = sub_trsv;
	cntl->sub_unpackv_x1 = sub_unpackv_x1;
}

