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

// NOTE: This is one of the few functions in BLIS that is defined
// with heterogeneous type support. This is done so that we have
// an operation that can be used to typecast (copy-cast) a scalar
// of one datatype to a scalar of another datatype.

typedef void (*FUNCPTR_T)
     (
             conj_t conjchi,
       const void*  chi,
             void*  psi
     );

static FUNCPTR_T GENARRAY2_ALL(ftypes,copysc);

//
// Define object-based interfaces.
//

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
	conj_t conjchi = bli_obj_conj_status( chi ); \
\
	num_t  dt_psi  = bli_obj_dt( psi ); \
	void*  buf_psi = bli_obj_buffer_at_off( psi ); \
\
	num_t  dt_chi; \
	void*  buf_chi; \
\
	FUNCPTR_T f; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, psi ); \
\
	/* If chi is a scalar constant, use dt_psi to extract the address of the
	   corresponding constant value; otherwise, use the datatype encoded
	   within the chi object and extract the buffer at the chi offset. */ \
	bli_obj_scalar_set_dt_buffer( chi, dt_psi, &dt_chi, &buf_chi ); \
\
	/* Index into the type combination array to extract the correct
	   function pointer. */ \
	f = ftypes[dt_chi][dt_psi]; \
\
	/* Invoke the void pointer-based function. */ \
	f( \
	   conjchi, \
	   buf_chi, \
	   buf_psi  \
	 ); \
}

GENFRONT( copysc )


//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_y, chx, chy, varname ) \
\
void PASTEMAC(chx,chy,varname) \
     ( \
             conj_t conjchi, \
       const void*  chi, \
             void*  psi \
     ) \
{ \
	bli_init_once(); \
\
	const ctype_x* chi_cast = chi; \
	      ctype_y* psi_cast = psi; \
\
	if ( bli_is_conj( conjchi ) ) \
	{ \
		bli_tcopyjs( chx,chy, *chi_cast, *psi_cast ); \
	} \
	else \
	{ \
		bli_tcopys( chx,chy, *chi_cast, *psi_cast ); \
	} \
}

INSERT_GENTFUNC2_BASIC( copysc )
INSERT_GENTFUNC2_MIX_DP( copysc )

