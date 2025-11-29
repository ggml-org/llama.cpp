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

#ifndef BLIS_MACRO_DEFS_H
#define BLIS_MACRO_DEFS_H


// -- Concatenation macros --

#define BLIS_FUNC_PREFIX_STR       "bli"

// We add an extra layer the definitions of these string-pasting macros
// because sometimes it is needed if, for example, one of the PASTE
// macros is invoked with an "op" argument that is itself a macro.

#define PASTEMAC0_(op)                         bli_ ## op
#define PASTEMAC1_(ch,op)                      bli_ ## ch  ## op
#define PASTEMAC2_(ch1,ch2,op)                 bli_ ## ch1 ## ch2 ## op
#define PASTEMAC3_(ch1,ch2,ch3,op)             bli_ ## ch1 ## ch2 ## ch3 ## op
#define PASTEMAC4_(ch1,ch2,ch3,ch4,op)         bli_ ## ch1 ## ch2 ## ch3 ## ch4 ## op
#define PASTEMAC5_(ch1,ch2,ch3,ch4,ch5,op)     bli_ ## ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## op
#define PASTEMAC6_(ch1,ch2,ch3,ch4,ch5,ch6,op) bli_ ## ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## ch6 ## op

#define PASTEMAC__(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,...) PASTEMAC ## arg8 ## _
#define PASTEMAC_(...) PASTEMAC__(__VA_ARGS__, 6, 5, 4, 3, 2, 1, 0, XXX)
#define PASTEMAC(...) PASTEMAC_(__VA_ARGS__)(__VA_ARGS__)

#define PASTECH0_(op)                         op
#define PASTECH1_(ch,op)                      ch  ## op
#define PASTECH2_(ch1,ch2,op)                 ch1 ## ch2 ## op
#define PASTECH3_(ch1,ch2,ch3,op)             ch1 ## ch2 ## ch3 ## op
#define PASTECH4_(ch1,ch2,ch3,ch4,op)         ch1 ## ch2 ## ch3 ## ch4 ## op
#define PASTECH5_(ch1,ch2,ch3,ch4,ch5,op)     ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## op
#define PASTECH6_(ch1,ch2,ch3,ch4,ch5,ch6,op) ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## ch6 ## op

#define PASTECH__(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,...) PASTECH ## arg8 ## _
#define PASTECH_(...) PASTECH__(__VA_ARGS__, 6, 5, 4, 3, 2, 1, 0, XXX)
#define PASTECH(...) PASTECH_(__VA_ARGS__)(__VA_ARGS__)

// Fortran-77 name-mangling macros.
#define PASTEF770_(op)                         op  ## _
#define PASTEF771_(ch,op)                      ch  ## op ## _
#define PASTEF772_(ch1,ch2,op)                 ch1 ## ch2 ## op ## _
#define PASTEF773_(ch1,ch2,ch3,op)             ch1 ## ch2 ## ch3 ## op ## _
#define PASTEF774_(ch1,ch2,ch3,ch4,op)         ch1 ## ch2 ## ch3 ## ch4 ## op ## _
#define PASTEF775_(ch1,ch2,ch3,ch4,ch5,op)     ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## op ## _
#define PASTEF776_(ch1,ch2,ch3,ch4,ch5,ch6,op) ch1 ## ch2 ## ch3 ## ch4 ## ch5 ## ch6 ## op ## _

#define PASTEF77__(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,...) PASTEF77 ## arg8 ## _
#define PASTEF77_(...) PASTEF77__(__VA_ARGS__, 6, 5, 4, 3, 2, 1, 0, XXX)
#define PASTEF77(...) PASTEF77_(__VA_ARGS__)(__VA_ARGS__)

#define PASTEBLACHK_(op)           bla_ ## op ## _check
#define PASTEBLACHK(op)            PASTEBLACHK_(op)

#define MKSTR(s1)                  #s1
#define STRINGIFY_INT( s )         MKSTR( s )

// -- Include other groups of macros

#include "bli_genarray_macro_defs.h"
#include "bli_gentdef_macro_defs.h"
#include "bli_gentfunc_macro_defs.h"
#include "bli_gentprot_macro_defs.h"
#include "bli_gentconf_macro_defs.h"

#include "bli_misc_macro_defs.h"
#include "bli_cast_macro_defs.h"
#include "bli_edge_case_macro_defs.h"
#include "bli_param_macro_defs.h"
#include "bli_complex_macro_defs.h"
#include "bli_obj_macro_defs.h"
#include "bli_scalar_macro_defs.h"
#include "bli_error_macro_defs.h"
#include "bli_blas_macro_defs.h"
#include "bli_builtin_macro_defs.h"

#include "bli_oapi_macro_defs.h"
#include "bli_tapi_macro_defs.h"


#endif
