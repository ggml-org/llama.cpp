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
// -- Utility function types ---------------------------------------------------
//

// asumv

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             dim_t    n, \
       const ctype*   x, inc_t incx, \
             ctype_r* asum  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEFR( asumv )

// mkherm, mksymm, mktrim

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( mkherm )
INSERT_GENTDEF( mksymm )
INSERT_GENTDEF( mktrim )

// norm1v, normfv, normiv

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             dim_t    n, \
       const ctype*   x, inc_t incx, \
             ctype_r* norm  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEFR( norm1v )
INSERT_GENTDEFR( normfv )
INSERT_GENTDEFR( normiv )

// norm1m, normfm, normim

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
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

INSERT_GENTDEFR( norm1m )
INSERT_GENTDEFR( normfm )
INSERT_GENTDEFR( normim )

// fprintv

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             FILE*  file, \
       const char*  s1, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const char*  format, \
       const char*  s2  \
     );

INSERT_GENTDEF( fprintv )

// fprintm

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             FILE*  file, \
       const char*  s1, \
             dim_t  m, \
             dim_t  n, \
       const ctype* x, inc_t rs_x, inc_t cs_x, \
       const char*  format, \
       const char*  s2  \
     );

INSERT_GENTDEF( fprintm )

// randv, randnv

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
       dim_t   n, \
       ctype*  x, inc_t incx  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( randv )
INSERT_GENTDEF( randnv )

// randm, randnm

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
       doff_t  diagoffx, \
       uplo_t  uplox, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( randm )
INSERT_GENTDEF( randnm )

// sumsqv

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             dim_t    n, \
       const ctype*   x, inc_t incx, \
             ctype_r* scale, \
             ctype_r* sumsq  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEFR( sumsqv )

// -----------------------------------------------------------------------------

// Operations with only basic interfaces.

#ifdef BLIS_TAPI_BASIC

// eqsc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
       const ctype* psi, \
             bool*  is  \
     );

INSERT_GENTDEF( eqsc )

// eqv

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const ctype* y, inc_t incy, \
             bool*  is  \
     );

INSERT_GENTDEF( eqv )

// eqm

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
       const ctype*  y, inc_t rs_y, inc_t cs_y, \
             bool*   is  \
     );

INSERT_GENTDEF( eqm )

// ltsc, lesc, gtsc, gesc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype* chi, \
       const ctype* psi, \
             bool*  is  \
     );

INSERT_GENTDEF( ltsc )
INSERT_GENTDEF( lesc )
INSERT_GENTDEF( gtsc )
INSERT_GENTDEF( gesc )

#endif // #ifdef BLIS_OAPI_BASIC

