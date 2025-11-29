/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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

#ifndef BLIS_BLAS_DEFS_H
#define BLIS_BLAS_DEFS_H


// -- System headers needed by BLAS compatibility layer --

#include <ctype.h>  // for toupper(), used in xerbla()


// -- Constants --

#define BLIS_MAX_BLAS_FUNC_STR_LENGTH (7+1)


// -- Utility macros --

#include "bla_r_sign.h"
#include "bla_d_sign.h"

#include "bla_r_cnjg.h"
#include "bla_d_cnjg.h"

#include "bla_r_imag.h"
#include "bla_d_imag.h"

#include "bla_c_div.h"
#include "bla_z_div.h"

#include "bla_f__cabs.h" // needed by c_abs, z_abs
#include "bla_r_abs.h"
#include "bla_d_abs.h"
#include "bla_c_abs.h"
#include "bla_z_abs.h"

#include "bla_lsame.h"
#include "bla_xerbla.h"
#include "bla_xerbla_array.h"


// -- Level-0 BLAS prototypes --

#include "bla_cabs1.h"


// -- Level-1 BLAS prototypes --

#include "bla_amax.h"
#include "bla_asum.h"
#include "bla_axpy.h"
#include "bla_copy.h"
#include "bla_dot.h"
#include "bla_nrm2.h"
#include "bla_rot.h"
#include "bla_rotg.h"
#include "bla_rotm.h"
#include "bla_rotmg.h"
#include "bla_scal.h"
#include "bla_swap.h"

#include "f77_amax_sub.h"
#include "f77_asum_sub.h"
#include "f77_dot_sub.h"
#include "f77_nrm2_sub.h"


// -- Level-2 BLAS prototypes --

// dense

#include "bla_gemv.h"
#include "bla_ger.h"
#include "bla_hemv.h"
#include "bla_her.h"
#include "bla_her2.h"
#include "bla_symv.h"
#include "bla_syr.h"
#include "bla_syr2.h"
#include "bla_trmv.h"
#include "bla_trsv.h"

#include "bla_gemv_check.h"
#include "bla_ger_check.h"
#include "bla_hemv_check.h"
#include "bla_her_check.h"
#include "bla_her2_check.h"
#include "bla_symv_check.h"
#include "bla_syr_check.h"
#include "bla_syr2_check.h"
#include "bla_trmv_check.h"
#include "bla_trsv_check.h"

// packed

#include "bla_hpmv.h"
#include "bla_hpr.h"
#include "bla_hpr2.h"
#include "bla_spmv.h"
#include "bla_spr.h"
#include "bla_spr2.h"
#include "bla_tpmv.h"
#include "bla_tpsv.h"

// banded

#include "bla_gbmv.h"
#include "bla_hbmv.h"
#include "bla_sbmv.h"
#include "bla_tbmv.h"
#include "bla_tbsv.h"


// -- Level-3 BLAS prototypes --

#include "bla_gemm.h"
#include "bla_hemm.h"
#include "bla_herk.h"
#include "bla_her2k.h"
#include "bla_symm.h"
#include "bla_syrk.h"
#include "bla_syr2k.h"
#include "bla_trmm.h"
#include "bla_trsm.h"

#include "bla_gemm_check.h"
#include "bla_hemm_check.h"
#include "bla_herk_check.h"
#include "bla_her2k_check.h"
#include "bla_symm_check.h"
#include "bla_syrk_check.h"
#include "bla_syr2k_check.h"
#include "bla_trmm_check.h"
#include "bla_trsm_check.h"


// -- BLAS extension prototypes --

// unique to BLIS

#include "bla_axpby.h"

// level-3

#include "bla_gemmt.h"
#include "bla_gemmt_check.h"

// batch

#include "bla_gemm_batch.h"

// 3m

#include "bla_gemm3m.h"
#include "bla_gemm3m_check.h"


// -- Fortran-compatible APIs to BLIS functions --

#include "b77_thread.h"


#endif // BLIS_BLAS_DEFS_H

