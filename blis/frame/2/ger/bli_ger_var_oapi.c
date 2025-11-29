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

#undef  GENFRONT
#define GENFRONT( opname, varname ) \
\
void PASTEMAC(varname) \
     ( \
       obj_t*  alpha, \
       obj_t*  x, \
       obj_t*  y, \
       obj_t*  a, \
       cntx_t* cntx, \
       cntl_t* cntl \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt        = bli_obj_dt( a ); \
\
	conj_t    conjx     = bli_obj_conj_status( x ); \
	conj_t    conjy     = bli_obj_conj_status( y ); \
\
	dim_t     m         = bli_obj_length( a ); \
	dim_t     n         = bli_obj_width( a ); \
\
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
\
	void*     buf_y     = bli_obj_buffer_at_off( y ); \
	inc_t     incy      = bli_obj_vector_inc( y ); \
\
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	inc_t     rs_a      = bli_obj_row_stride( a ); \
	inc_t     cs_a      = bli_obj_col_stride( a ); \
\
	void*     buf_alpha = bli_obj_buffer_for_1x1( dt, alpha ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_unb,_vft) f = \
	PASTEMAC(varname,_qfp)( dt ); \
\
	f \
	( \
	  conjx, \
	  conjy, \
	  m, \
	  n, \
	  buf_alpha, \
	  buf_x, incx, \
	  buf_y, incy, \
	  buf_a, rs_a, cs_a, \
	  cntx  \
	); \
} \

GENFRONT( ger, ger_unb_var1 )
GENFRONT( ger, ger_unb_var2 )

