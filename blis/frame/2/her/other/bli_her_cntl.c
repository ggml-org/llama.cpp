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
extern ger_t*     ger_cntl_bs_ke_row;
extern ger_t*     ger_cntl_bs_ke_col;

her_t*            her_cntl_bs_ke_lrow_ucol = NULL;
her_t*            her_cntl_bs_ke_lcol_urow = NULL;

her_t*            her_cntl_ge_lrow_ucol = NULL;
her_t*            her_cntl_ge_lcol_urow = NULL;


void bli_her_cntl_init()
{
	// Create control trees for the lowest-level kernels. These trees induce
	// operations on (persumably) relatively small block-subvector problems.
	her_cntl_bs_ke_lrow_ucol
	=
	bli_her_cntl_obj_create( BLIS_UNBLOCKED,
	                         BLIS_VARIANT1,
	                         0,
	                         NULL, NULL, NULL,
	                         NULL, NULL );
	her_cntl_bs_ke_lcol_urow
	=
	bli_her_cntl_obj_create( BLIS_UNBLOCKED,
	                         BLIS_VARIANT2,
	                         0,
	                         NULL, NULL, NULL,
	                         NULL, NULL );


	// Create control trees for generally large problems. Here, we choose
	// variants that partition for ger subproblems in the same direction
	// as the assumed storage.
	her_cntl_ge_lrow_ucol
	=
	bli_her_cntl_obj_create( BLIS_BLOCKED,
	                         BLIS_VARIANT1,
	                         BLIS_M2,
	                         packv_cntl,       // pack x1 (if needed)
	                         NULL,             // do NOT pack C11
	                         ger_cntl_rp_bs_row,
	                         her_cntl_bs_ke_lrow_ucol,
	                         NULL );           // no unpacking needed
	her_cntl_ge_lcol_urow
	=
	bli_her_cntl_obj_create( BLIS_BLOCKED,
	                         BLIS_VARIANT2,
	                         BLIS_M2,
	                         packv_cntl,       // pack x1 (if needed)
	                         NULL,             // do NOT pack C11
	                         ger_cntl_cp_bs_col,
	                         her_cntl_bs_ke_lcol_urow,
	                         NULL );           // no unpacking needed
}

void bli_her_cntl_finalize()
{
	bli_cntl_free_node( her_cntl_bs_ke_lrow_ucol );
	bli_cntl_free_node( her_cntl_bs_ke_lcol_urow );
	bli_cntl_free_node( her_cntl_ge_lrow_ucol );
	bli_cntl_free_node( her_cntl_ge_lcol_urow );
}


her_t* bli_her_cntl_obj_create( impl_t     impl_type,
                                varnum_t   var_num,
                                bszid_t    bszid,
                                packv_t*   sub_packv_x1,
                                packm_t*   sub_packm_c11,
                                ger_t*     sub_ger,
                                her_t*     sub_her,
                                unpackm_t* sub_unpackm_c11 )
{
	her_t* cntl;

	cntl = ( her_t* ) bli_malloc_intl( sizeof(her_t) );

	cntl->impl_type       = impl_type;
	cntl->var_num         = var_num;
	cntl->bszid           = bszid;
	cntl->sub_packv_x1    = sub_packv_x1;
	cntl->sub_packm_c11   = sub_packm_c11;
	cntl->sub_ger         = sub_ger;
	cntl->sub_her         = sub_her;
	cntl->sub_unpackm_c11 = sub_unpackm_c11;

	return cntl;
}

void bli_her_cntl_obj_init( her_t*     cntl,
                            impl_t     impl_type,
                            varnum_t   var_num,
                            bszid_t    bszid,
                            packv_t*   sub_packv_x1,
                            packm_t*   sub_packm_c11,
                            ger_t*     sub_ger,
                            her_t*     sub_her,
                            unpackm_t* sub_unpackm_c11 )
{
	cntl->impl_type       = impl_type;
	cntl->var_num         = var_num;
	cntl->bszid           = bszid;
	cntl->sub_packv_x1    = sub_packv_x1;
	cntl->sub_packm_c11   = sub_packm_c11;
	cntl->sub_ger         = sub_ger;
	cntl->sub_her         = sub_her;
	cntl->sub_unpackm_c11 = sub_unpackm_c11;
}


