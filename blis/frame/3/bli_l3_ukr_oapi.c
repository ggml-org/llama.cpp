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
#define GENFRONT( tname, opname ) \
\
void PASTEMAC(opname) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt        = bli_obj_dt( c ); \
\
	dim_t     m         = bli_obj_length( c ); \
	dim_t     n         = bli_obj_width( c ); \
	dim_t     k         = bli_obj_width( a ); \
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	void*     buf_b     = bli_obj_buffer_at_off( b ); \
	void*     buf_c     = bli_obj_buffer_at_off( c ); \
	inc_t     rs_c      = bli_obj_row_stride( c ); \
	inc_t     cs_c      = bli_obj_col_stride( c ); \
	void*     buf_alpha = bli_obj_buffer_for_1x1( dt, alpha ); \
	void*     buf_beta  = bli_obj_buffer_for_1x1( dt, beta ); \
\
	auxinfo_t data; \
\
	/* Fill the auxinfo_t struct in case the micro-kernel uses it. */ \
	bli_auxinfo_set_next_a( buf_a, &data ); \
	bli_auxinfo_set_next_b( buf_b, &data ); \
	bli_auxinfo_set_is_a( 1, &data ); \
	bli_auxinfo_set_is_b( 1, &data ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(tname,_ukr_ft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  m, \
	  n, \
	  k, \
	  buf_alpha, \
	  buf_a, \
	  buf_b, \
	  buf_beta, \
	  buf_c, rs_c, cs_c, \
	  &data, \
	  cntx  \
	); \
} \

GENFRONT( gemm, gemm_ukernel )


#undef  GENFRONT
#define GENFRONT( tname, opname, opnamel, opnameu ) \
\
void PASTEMAC(opname) \
     ( \
       obj_t*  alpha, \
       obj_t*  a1x, \
       obj_t*  a11, \
       obj_t*  bx1, \
       obj_t*  b11, \
       obj_t*  c11, \
       cntx_t* cntx  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt        = bli_obj_dt( c11 ); \
\
	dim_t     m         = bli_obj_length( c11 ); \
	dim_t     n         = bli_obj_width( c11 ); \
	dim_t     k         = bli_obj_width( a1x ); \
	void*     buf_a1x   = bli_obj_buffer_at_off( a1x ); \
	void*     buf_a11   = bli_obj_buffer_at_off( a11 ); \
	void*     buf_bx1   = bli_obj_buffer_at_off( bx1 ); \
	void*     buf_b11   = bli_obj_buffer_at_off( b11 ); \
	void*     buf_c11   = bli_obj_buffer_at_off( c11 ); \
	inc_t     rs_c      = bli_obj_row_stride( c11 ); \
	inc_t     cs_c      = bli_obj_col_stride( c11 ); \
	void*     buf_alpha = bli_obj_buffer_for_1x1( dt, alpha ); \
\
	auxinfo_t data; \
\
	/* Fill the auxinfo_t struct in case the micro-kernel uses it. */ \
	if ( bli_obj_is_lower( a11 ) ) \
	{ bli_auxinfo_set_next_a( buf_a1x, &data ); } \
	else /* if ( bli_obj_is_upper( a11 ) ) */ \
	{ bli_auxinfo_set_next_a( buf_a11, &data ); } \
	bli_auxinfo_set_next_b( buf_bx1, &data ); \
\
	/* Invoke the void pointer-based function for the given datatype. */ \
	if ( bli_obj_is_lower( a11 ) ) \
	{ \
		/* Query a type-specific function pointer, except one that uses
		   void* for function arguments instead of typed pointers. */ \
		PASTECH(tname,_ukr_ft) f = \
		PASTEMAC(opnamel,_qfp)( dt ); \
\
		f \
		( \
		  m, \
		  n, \
		  k, \
		  buf_alpha, \
		  buf_a1x, \
		  buf_a11, \
		  buf_bx1, \
		  buf_b11, \
		  buf_c11, rs_c, cs_c, \
		  &data, \
		  cntx  \
		); \
	} \
	else /* if ( bli_obj_is_upper( a11 ) ) */ \
	{ \
		/* Query a type-specific function pointer, except one that uses
		   void* for function arguments instead of typed pointers. */ \
		PASTECH(tname,_ukr_ft) f = \
		PASTEMAC(opnameu,_qfp)( dt ); \
\
		f \
		( \
		  m, \
		  n, \
		  k, \
		  buf_alpha, \
		  buf_a1x, \
		  buf_a11, \
		  buf_bx1, \
		  buf_b11, \
		  buf_c11, rs_c, cs_c, \
		  &data, \
		  cntx  \
		); \
	} \
} \

GENFRONT( gemmtrsm, gemmtrsm_ukernel, gemmtrsm_l_ukernel, gemmtrsm_u_ukernel )


#undef  GENFRONT
#define GENFRONT( tname, opname, opnamel, opnameu ) \
\
void PASTEMAC(opname) \
     ( \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  c, \
       cntx_t* cntx  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt        = bli_obj_dt( c ); \
\
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	void*     buf_b     = bli_obj_buffer_at_off( b ); \
	void*     buf_c     = bli_obj_buffer_at_off( c ); \
	inc_t     rs_c      = bli_obj_row_stride( c ); \
	inc_t     cs_c      = bli_obj_col_stride( c ); \
\
	auxinfo_t data; \
\
	/* Fill the auxinfo_t struct in case the micro-kernel uses it. */ \
	bli_auxinfo_set_next_a( buf_a, &data ); \
	bli_auxinfo_set_next_b( buf_b, &data ); \
	bli_auxinfo_set_is_a( 1, &data ); \
	bli_auxinfo_set_is_b( 1, &data ); \
\
	/* Invoke the void pointer-based function for the given datatype. */ \
	if ( bli_obj_is_lower( a ) ) \
	{ \
		/* Query a type-specific function pointer, except one that uses
		   void* for function arguments instead of typed pointers. */ \
		PASTECH(tname,_ukr_ft) f = \
		PASTEMAC(opnamel,_qfp)( dt ); \
\
		f \
		( \
		  buf_a, \
		  buf_b, \
		  buf_c, rs_c, cs_c, \
		  &data, \
		  cntx  \
		); \
	} \
	else /* if ( bli_obj_is_upper( a ) ) */ \
	{ \
		/* Query a type-specific function pointer, except one that uses
		   void* for function arguments instead of typed pointers. */ \
		PASTECH(tname,_ukr_ft) f = \
		PASTEMAC(opnameu,_qfp)( dt ); \
\
		f \
		( \
		  buf_a, \
		  buf_b, \
		  buf_c, rs_c, cs_c, \
		  &data, \
		  cntx  \
		); \
	} \
} \

GENFRONT( trsm, trsm_ukernel, trsm_l_ukernel, trsm_u_ukernel )

