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

#include "bli_l2_check.h"

// Define function types.
#include "bli_l2_ft_unb.h"

// Prototype object APIs (expert and non-expert).
#include "bli_oapi_ex.h"
#include "bli_l2_oapi.h"
#include "bli_xapi_undef.h"

#include "bli_oapi_ba.h"
#include "bli_l2_oapi.h"
#include "bli_xapi_undef.h"

// Prototype typed APIs (expert and non-expert).
#include "bli_tapi_ex.h"
#include "bli_l2_tapi.h"
#include "bli_l2_ft.h"
#include "bli_xapi_undef.h"

#include "bli_tapi_ba.h"
#include "bli_l2_tapi.h"
#include "bli_l2_ft.h"
#include "bli_xapi_undef.h"

// Generate function pointer arrays for tapi functions (expert only).
#include "bli_l2_fpa.h"

// Operation-specific headers
#include "bli_gemv.h"
#include "bli_ger.h"
#include "bli_hemv.h"
#include "bli_her.h"
#include "bli_her2.h"
#include "bli_symv.h"
#include "bli_syr.h"
#include "bli_syr2.h"
#include "bli_trmv.h"
#include "bli_trsv.h"

