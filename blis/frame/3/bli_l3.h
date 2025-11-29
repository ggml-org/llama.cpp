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

#include "bli_l3_util.h"
#include "bli_l3_thrinfo.h"
#include "bli_l3_decor.h"
#include "bli_l3_sup_decor.h"

#include "bli_l3_check.h"
#include "bli_l3_packab.h"

// Define function types.
#include "bli_l3_ukr_ft.h"
#include "bli_l3_oft.h"
#include "bli_l3_oft_var.h"

#include "bli_l3_int.h"
#include "bli_l3_prune.h"

// Prototype object APIs (basic and expert).
#include "bli_l3_oapi.h"
#include "bli_l3_oapi_ex.h"

// Prototype typed APIs (basic and expert).
#include "bli_l3_tapi.h"
#include "bli_l3_tapi_ex.h"

// Define function types for small/unpacked handlers/kernels.
#include "bli_l3_sup_oft.h"
#include "bli_l3_sup_ker_ft.h"

// Define static edge case logic for use in small/unpacked kernels.
//#include "bli_l3_sup_edge.h"

// Prototype object API to small/unpacked matrix dispatcher.
#include "bli_l3_sup.h"

// Prototype reference implementation of small/unpacked matrix handler.
#include "bli_l3_sup_ref.h"
#include "bli_l3_sup_int.h"
#include "bli_l3_sup_vars.h"
#include "bli_l3_sup_packm.h"
#include "bli_l3_sup_packm_var.h"

// Prototype microkernel wrapper APIs.
#include "bli_l3_ukr_oapi.h"
#include "bli_l3_ukr_tapi.h"

// Generate function pointer arrays for tapi microkernel functions.
#include "bli_l3_ukr_fpa.h"

// Operation-specific headers.
#include "bli_gemm.h"
#include "bli_trmm.h"
#include "bli_trsm.h"
#include "bli_gemmt.h"
