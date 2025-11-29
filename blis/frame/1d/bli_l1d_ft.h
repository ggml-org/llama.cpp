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
// -- Level-1d function types --------------------------------------------------
//

// addd, copyd, subd

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( addd )
INSERT_GENTDEF( copyd )
INSERT_GENTDEF( subd )

// axpyd, scal2d

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( axpyd )
INSERT_GENTDEF( scal2d )

// invertd

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
       doff_t diagoffx, \
       dim_t  m, \
       dim_t  n, \
       ctype* x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( invertd )

// invscald, scald, setd

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             conj_t conjalpha, \
             doff_t diagoffx, \
             dim_t  m, \
             dim_t  n, \
       const ctype* alpha, \
             ctype* x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( invscald )
INSERT_GENTDEF( scald )
INSERT_GENTDEF( setd )

// setid

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             doff_t   diagoffx, \
             dim_t    m, \
             dim_t    n, \
       const ctype_r* alpha, \
             ctype*   x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEFR( setid )

// shiftd

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             doff_t diagoffx, \
             dim_t  m, \
             dim_t  n, \
       const ctype* alpha, \
             ctype* x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( shiftd )

// xpbyd

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,EX_SUF,tsuf)) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
       const ctype*  beta, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTDEF( xpbyd )

