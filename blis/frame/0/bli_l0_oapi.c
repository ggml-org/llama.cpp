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

//
// Define object-based interfaces.
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* chi, \
       const obj_t* absq  \
     ) \
{ \
	bli_init_once(); \
\
	num_t       dt_chi; \
	num_t       dt_absq_c  = bli_obj_dt_proj_to_complex( absq ); \
\
	const void* buf_chi; \
	void*       buf_absq   = bli_obj_buffer_at_off( absq ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, absq ); \
\
	/* If chi is a scalar constant, use dt_absq_c to extract the address of the
	   corresponding constant value; otherwise, use the datatype encoded
	   within the chi object and extract the buffer at the chi offset. */ \
	bli_obj_scalar_set_dt_buffer( chi, dt_absq_c, &dt_chi, ( void** )&buf_chi ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = PASTEMAC(opname,_qfp)( dt_chi ); \
\
	f \
	( \
	  buf_chi, \
	  buf_absq  \
	); \
}

GENFRONT( absqsc )
GENFRONT( normfsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* chi, \
       const obj_t* psi  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt        = bli_obj_dt( psi ); \
\
	conj_t    conjchi   = bli_obj_conj_status( chi ); \
\
	void*     buf_chi   = bli_obj_buffer_for_1x1( dt, chi ); \
	void*     buf_psi   = bli_obj_buffer_at_off( psi ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, psi ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  conjchi, \
	  buf_chi, \
	  buf_psi  \
	); \
}

GENFRONT( addsc )
GENFRONT( divsc )
GENFRONT( mulsc )
GENFRONT( subsc )
GENFRONT( invertsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* chi, \
       const obj_t* psi  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt        = bli_obj_dt( psi ); \
\
	void*     buf_chi   = bli_obj_buffer_for_1x1( dt, chi ); \
	void*     buf_psi   = bli_obj_buffer_at_off( psi ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, psi ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  buf_chi, \
	  buf_psi  \
	); \
}

GENFRONT( sqrtsc )
GENFRONT( sqrtrsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t*  chi, \
             double* zeta_r, \
             double* zeta_i  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt_chi    = bli_obj_dt( chi ); \
	num_t     dt_def    = BLIS_DCOMPLEX; \
	num_t     dt_use; \
\
	/* If chi is a constant object, default to using the dcomplex
	   value to maximize precision, and since we don't know if the
	   caller needs just the real or the real and imaginary parts. */ \
	void*     buf_chi   = bli_obj_buffer_for_1x1( dt_def, chi ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, zeta_r, zeta_i ); \
\
	/* The _check() routine prevents integer types, so we know that chi
	   is either a constant or an actual floating-point type. */ \
	if ( bli_is_constant( dt_chi ) ) dt_use = dt_def; \
	else                             dt_use = dt_chi; \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = PASTEMAC(opname,_qfp)( dt_use ); \
\
	f \
	( \
	  buf_chi, \
	  zeta_r, \
	  zeta_i  \
	); \
}

GENFRONT( getsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
             double zeta_r, \
             double zeta_i, \
       const obj_t* chi  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt_chi    = bli_obj_dt( chi ); \
\
	void*     buf_chi   = bli_obj_buffer_at_off( chi ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( zeta_r, zeta_i, chi ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = PASTEMAC(opname,_qfp)( dt_chi ); \
\
	f \
	( \
	  zeta_r, \
	  zeta_i, \
	  buf_chi  \
	); \
}

GENFRONT( setsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* chi, \
       const obj_t* zeta_r, \
       const obj_t* zeta_i  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt_chi; \
	num_t     dt_zeta_c   = bli_obj_dt_proj_to_complex( zeta_r ); \
\
	void*     buf_chi; \
\
	void*     buf_zeta_r  = bli_obj_buffer_at_off( zeta_r ); \
	void*     buf_zeta_i  = bli_obj_buffer_at_off( zeta_i ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, zeta_r, zeta_i ); \
\
	/* If chi is a scalar constant, use dt_zeta_c to extract the address of the
	   corresponding constant value; otherwise, use the datatype encoded
	   within the chi object and extract the buffer at the chi offset. */ \
	bli_obj_scalar_set_dt_buffer( chi, dt_zeta_c, &dt_chi, &buf_chi ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = PASTEMAC(opname,_qfp)( dt_chi ); \
\
	f \
	( \
	  buf_chi, \
	  buf_zeta_r, \
	  buf_zeta_i  \
	); \
}

GENFRONT( unzipsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* zeta_r, \
       const obj_t* zeta_i, \
       const obj_t* chi  \
     ) \
{ \
	bli_init_once(); \
\
	num_t     dt_chi      = bli_obj_dt( chi ); \
\
	void*     buf_zeta_r  = bli_obj_buffer_for_1x1( dt_chi, zeta_r ); \
	void*     buf_zeta_i  = bli_obj_buffer_for_1x1( dt_chi, zeta_i ); \
\
	void*     buf_chi     = bli_obj_buffer_at_off( chi ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, zeta_r, zeta_i ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = PASTEMAC(opname,_qfp)( dt_chi ); \
\
	f \
	( \
	  buf_zeta_i, \
	  buf_zeta_r, \
	  buf_chi  \
	); \
}

GENFRONT( zipsc )

