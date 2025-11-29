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


//
// Prototype BLAS-like interfaces with typed operands.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
      ( \
              conj_t conjx, \
              dim_t  n, \
        const ctype* x, inc_t incx, \
              ctype* y, inc_t incy  \
        BLIS_TAPI_EX_PARAMS  \
      );

INSERT_GENTPROT_BASIC( addv )
INSERT_GENTPROT_BASIC( copyv )
INSERT_GENTPROT_BASIC( subv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             dim_t  n, \
       const ctype* x, inc_t incx, \
             dim_t* index  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( amaxv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* beta, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( axpbyv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( axpyv )
INSERT_GENTPROT_BASIC( scal2v )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             ctype* rho  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( dotv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             conj_t conjy, \
             dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
       const ctype* beta, \
             ctype* rho  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( dotxv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       dim_t  n, \
       ctype* x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( invertv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjalpha, \
             dim_t  n, \
       const ctype* alpha, \
             ctype* x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( invscalv )
INSERT_GENTPROT_BASIC( scalv )
INSERT_GENTPROT_BASIC( setv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       dim_t  n, \
       ctype* x, inc_t incx, \
       ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( swapv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const ctype* beta, \
             ctype* y, inc_t incy  \
       BLIS_TAPI_EX_PARAMS  \
     ); \

INSERT_GENTPROT_BASIC( xpbyv )
