/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
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


void bli_packm_cntl_init_node
     (
       void_fp       var_func,
       packm_var_oft var,
       const void*   params,
       packm_cntl_t* cntl
     )
{
	// Initialize the packm_cntl_t struct.
	cntl->var    = var;
	cntl->params = params;

	bli_cntl_init_node
	(
	  var_func,
	  &cntl->cntl
	);
}

void bli_packm_def_cntl_init_node
     (
       void_fp           var_func,
       num_t             dt_orig,
       num_t             dt_pack,
       num_t             dt_bmult,
       packm_ker_ft      ukr,
       dim_t             bmult_m_def,
       dim_t             bmult_m_pack,
       dim_t             bmult_m_bcast,
       dim_t             bmult_m_scale,
       dim_t             bmult_m_pack_scale,
       dim_t             bmult_n_def,
       bool              does_invert_diag,
       bool              rev_iter_if_upper,
       bool              rev_iter_if_lower,
       pack_t            pack_schema,
       packbuf_t         pack_buf_type,
       packm_def_cntl_t* cntl
     )
{
	// Initialize the packm_def_cntl_t struct.
	cntl->ukr                = ukr;
	cntl->dt_orig            = dt_orig;
	cntl->dt_pack            = dt_pack;
	cntl->dt_bmult           = dt_bmult;
	cntl->bmult_m_def        = bmult_m_def;
	cntl->bmult_m_pack       = bmult_m_pack;
	cntl->bmult_m_bcast      = bmult_m_bcast;
	cntl->bmult_m_scale      = bmult_m_scale;
	cntl->bmult_m_pack_scale = bmult_m_pack_scale;
	cntl->bmult_n_def        = bmult_n_def;
	cntl->does_invert_diag   = does_invert_diag;
	cntl->rev_iter_if_upper  = rev_iter_if_upper;
	cntl->rev_iter_if_lower  = rev_iter_if_lower;
	cntl->pack_schema        = pack_schema;
	cntl->pack_buf_type      = pack_buf_type;
	cntl->params             = cntl;

	bli_packm_cntl_init_node
	(
	  var_func,
	  bli_packm_blk_var1,
	  NULL,
	  &cntl->cntl
	);
}

