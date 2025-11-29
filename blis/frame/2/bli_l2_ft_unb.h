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

#ifndef BLIS_L2_FT_UNB_H
#define BLIS_L2_FT_UNB_H


//
// -- Level-2 function types ---------------------------------------------------
//

// gemv

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,_unb,tsuf)) \
     ( \
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     );

INSERT_GENTDEF( gemv )

// ger

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,_unb,tsuf)) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     );

INSERT_GENTDEF( ger )

// hemv (and symv)

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,_unb,tsuf)) \
     ( \
       uplo_t  uploa, \
       conj_t  conja, \
       conj_t  conjx, \
       conj_t  conjh, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     );

INSERT_GENTDEF( hemv )

// her (and syr)

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,_unb,tsuf)) \
     ( \
       uplo_t   uploa, \
       conj_t   conjx, \
       conj_t   conjh, \
       dim_t    m, \
       ctype*   alpha, /* complex alpha allows her variants to also perform syr. */ \
       ctype*   x, inc_t incx, \
       ctype*   a, inc_t rs_a, inc_t cs_a, \
       cntx_t*  cntx  \
     );

INSERT_GENTDEFR( her )

// her2 (and syr2)

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,_unb,tsuf)) \
     ( \
       uplo_t  uploa, \
       conj_t  conjx, \
       conj_t  conjy, \
       conj_t  conjh, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     );

INSERT_GENTDEF( her2 )

// trmv (and trsv)

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,_unb,tsuf)) \
     ( \
       uplo_t  uploa, \
       trans_t transa, \
       diag_t  diaga, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       cntx_t* cntx  \
     );

INSERT_GENTDEF( trmv )
INSERT_GENTDEF( trsv )


#endif
