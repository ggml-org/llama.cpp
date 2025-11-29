/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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
// Define object-based interfaces (basic).
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* b, \
       const obj_t* beta, \
       const obj_t* c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(opname,_ex)( alpha, a, b, beta, c, NULL, NULL ); \
}

GENFRONT( gemm )
GENFRONT( gemmt )
GENFRONT( her2k )
GENFRONT( syr2k )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
             side_t side, \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* b, \
       const obj_t* beta, \
       const obj_t* c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(opname,_ex)( side, alpha, a, b, beta, c, NULL, NULL ); \
}

GENFRONT( hemm )
GENFRONT( symm )
GENFRONT( trmm3 )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* beta, \
       const obj_t* c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(opname,_ex)( alpha, a, beta, c, NULL, NULL ); \
}

GENFRONT( herk )
GENFRONT( syrk )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
             side_t side, \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* b  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(opname,_ex)( side, alpha, a, b, NULL, NULL ); \
}

GENFRONT( trmm )
GENFRONT( trsm )

