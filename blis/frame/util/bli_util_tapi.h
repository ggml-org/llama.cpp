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

#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             dim_t    n, \
       const ctype*   x, inc_t incx, \
             ctype_r* asum  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROTR_BASIC( asumv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       uplo_t uploa, \
       dim_t  m, \
       ctype* a, inc_t rs_a, inc_t cs_a  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT_BASIC( mkherm )
INSERT_GENTPROT_BASIC( mksymm )
INSERT_GENTPROT_BASIC( mktrim )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             dim_t    n, \
       const ctype*   x, inc_t incx, \
             ctype_r* norm  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROTR_BASIC( norm1v )
INSERT_GENTPROTR_BASIC( normfv )
INSERT_GENTPROTR_BASIC( normiv )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t   diagoffx, \
             diag_t   diagx, \
             uplo_t   uplox, \
             dim_t    m, \
             dim_t    n, \
       const ctype*   x, inc_t rs_x, inc_t cs_x, \
             ctype_r* norm  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROTR_BASIC( norm1m )
INSERT_GENTPROTR_BASIC( normfm )
INSERT_GENTPROTR_BASIC( normim )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       dim_t  n, \
       ctype* x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT_BASIC( randv )
INSERT_GENTPROT_BASIC( randnv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
       doff_t diagoffx, \
       uplo_t uplox, \
       dim_t  m, \
       dim_t  n, \
       ctype* x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT_BASIC( randm )
INSERT_GENTPROT_BASIC( randnm )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             dim_t    n, \
       const ctype*   x, inc_t incx, \
             ctype_r* scale, \
             ctype_r* sumsq  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROTR_BASIC( sumsqv )

// -----------------------------------------------------------------------------

// Operations with basic interfaces only.

#ifdef BLIS_TAPI_BASIC

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
       const ctype* psi, \
             bool*  is_eq  \
     );

INSERT_GENTPROT_BASIC( eqsc )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
      ( \
              conj_t conjx, \
              dim_t  n, \
        const ctype* x, inc_t incx, \
        const ctype* y, inc_t incy, \
              bool*  is_eq  \
      );

INSERT_GENTPROT_BASIC( eqv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
       const ctype*  y, inc_t rs_y, inc_t cs_y, \
             bool*   is_eq  \
     );

INSERT_GENTPROT_BASIC( eqm )


#undef  GENTPROTRO
#define GENTPROTRO( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
       const ctype* chi, \
       const ctype* psi, \
             bool*  is  \
     );

INSERT_GENTPROTRO_BASIC( ltsc )
INSERT_GENTPROTRO_BASIC( lesc )
INSERT_GENTPROTRO_BASIC( gtsc )
INSERT_GENTPROTRO_BASIC( gesc )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
       const char* s1, \
             dim_t n, \
       const void* x, inc_t incx, \
       const char* format, \
       const char* s2  \
     );

INSERT_GENTPROT_BASIC_I( printv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
       const char* s1, \
             dim_t m, \
             dim_t n, \
       const void* x, inc_t rs_x, inc_t cs_x, \
       const char* format, \
       const char* s2  \
     );

INSERT_GENTPROT_BASIC_I( printm )

#endif // #ifdef BLIS_TAPI_BASIC

