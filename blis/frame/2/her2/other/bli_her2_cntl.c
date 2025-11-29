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

extern packm_t*   packm_cntl;
extern packv_t*   packv_cntl;
extern unpackm_t* unpackm_cntl;

extern ger_t*     ger_cntl_rp_bs_row;
extern ger_t*     ger_cntl_cp_bs_col;

her2_t*           her2_cntl_bs_ke_lrow_ucol = NULL;
her2_t*           her2_cntl_bs_ke_lcol_urow = NULL;

her2_t*           her2_cntl_ge_lrow_ucol = NULL;
her2_t*           her2_cntl_ge_lcol_urow = NULL;


void bli_her2_cntl_init()
{
	// Create control trees for the lowest-level kernels. These trees induce
    // operations on (persumably) relatively small block-subvector problems.
	her2_cntl_bs_ke_lrow_ucol
	=
	bli_her2_cntl_obj_create( BLIS_UNB_FUSED,
	                          BLIS_VARIANT1,
	                          0,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL,
	                          NULL );
	her2_cntl_bs_ke_lcol_urow
	=
	bli_her2_cntl_obj_create( BLIS_UNB_FUSED,
	                          BLIS_VARIANT4,
	                          0,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL,
	                          NULL );


	// Create control trees for generally large problems. Here, we choose
	// variants that partition for ger subproblems in the same direction
	// as the assumed storage.
	her2_cntl_ge_lrow_ucol
	=
	bli_her2_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          BLIS_M2,
	                          packv_cntl,     // pack x1 (if needed)
	                          packv_cntl,     // pack y1 (if needed)
	                          packm_cntl,     // pack C11 (if needed)
	                          ger_cntl_rp_bs_row,
	                          ger_cntl_rp_bs_row,
	                          her2_cntl_bs_ke_lrow_ucol,
	                          unpackm_cntl ); // unpack C11 (if packed)
	her2_cntl_ge_lcol_urow
	=
	bli_her2_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT4,
	                          BLIS_M2,
	                          packv_cntl,     // pack x1 (if needed)
	                          packv_cntl,     // pack y1 (if needed)
	                          packm_cntl,     // pack C11 (if needed)
	                          ger_cntl_cp_bs_col,
	                          ger_cntl_cp_bs_col,
	                          her2_cntl_bs_ke_lcol_urow,
	                          unpackm_cntl ); // unpack C11 (if packed)
}

void bli_her2_cntl_finalize()
{
	bli_cntl_free_node( her2_cntl_bs_ke_lrow_ucol );
	bli_cntl_free_node( her2_cntl_bs_ke_lcol_urow );
	bli_cntl_free_node( her2_cntl_ge_lrow_ucol );
	bli_cntl_free_node( her2_cntl_ge_lcol_urow );
}


her2_t* bli_her2_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  bszid_t    bszid,
                                  packv_t*   sub_packv_x1,
                                  packv_t*   sub_packv_y1,
                                  packm_t*   sub_packm_c11,
                                  ger_t*     sub_ger_rp,
                                  ger_t*     sub_ger_cp,
                                  her2_t*    sub_her2,
                                  unpackm_t* sub_unpackm_c11 )
{
	her2_t* cntl;

	cntl = ( her2_t* ) bli_malloc_intl( sizeof(her2_t) );

	cntl->impl_type       = impl_type;
	cntl->var_num         = var_num;
	cntl->bszid           = bszid;
	cntl->sub_packv_x1    = sub_packv_x1;
	cntl->sub_packv_y1    = sub_packv_y1;
	cntl->sub_packm_c11   = sub_packm_c11;
	cntl->sub_ger_rp      = sub_ger_rp;
	cntl->sub_ger_cp      = sub_ger_cp;
	cntl->sub_her2        = sub_her2;
	cntl->sub_unpackm_c11 = sub_unpackm_c11;

	return cntl;
}

void bli_her2_cntl_obj_init( her2_t*    cntl,
                             impl_t     impl_type,
                             varnum_t   var_num,
                             bszid_t    bszid,
                             packv_t*   sub_packv_x1,
                             packv_t*   sub_packv_y1,
                             packm_t*   sub_packm_c11,
                             ger_t*     sub_ger_rp,
                             ger_t*     sub_ger_cp,
                             her2_t*    sub_her2,
                             unpackm_t* sub_unpackm_c11 )
{
	cntl->impl_type       = impl_type;
	cntl->var_num         = var_num;
	cntl->bszid           = bszid;
	cntl->sub_packv_x1    = sub_packv_x1;
	cntl->sub_packv_y1    = sub_packv_y1;
	cntl->sub_packm_c11   = sub_packm_c11;
	cntl->sub_ger_rp      = sub_ger_rp;
	cntl->sub_ger_cp      = sub_ger_cp;
	cntl->sub_her2        = sub_her2;
	cntl->sub_unpackm_c11 = sub_unpackm_c11;
}


