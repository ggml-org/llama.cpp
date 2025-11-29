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
// -- Level-0 function types ---------------------------------------------------
//

// addsc, divsc, subsc, invertsc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
             ctype* psi  \
     );

INSERT_GENTDEF( addsc )
INSERT_GENTDEF( divsc )
INSERT_GENTDEF( subsc )
INSERT_GENTDEF( invertsc )

// mulsc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
             conj_t conjchi, \
       const ctype* chi, \
             ctype* psi  \
     );

INSERT_GENTDEF( mulsc )

// absqsc

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype*   chi, \
             ctype_r* absq  \
     );

INSERT_GENTDEFR( absqsc )

// normfsc

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype*   chi, \
             ctype_r* norm  \
     );

INSERT_GENTDEFR( normfsc )

// sqrtsc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype* chi, \
             ctype* psi  \
     );

INSERT_GENTDEF( sqrtsc )

// sqrtrsc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype* chi, \
             ctype* psi  \
     );

INSERT_GENTDEF( sqrtrsc )

// getsc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype*  chi, \
             double* zeta_r, \
             double* zeta_i  \
     );

INSERT_GENTDEF( getsc )

// setsc

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       double  zeta_r, \
       double  zeta_i, \
       ctype*  chi  \
     );

INSERT_GENTDEF( setsc )

// unzipsc

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype*   chi, \
             ctype_r* zeta_r, \
             ctype_r* zeta_i  \
     );

INSERT_GENTDEFR( unzipsc )

// zipsc

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH(ch,opname,tsuf)) \
     ( \
       const ctype_r* zeta_r, \
       const ctype_r* zeta_i, \
             ctype*   chi  \
     );

INSERT_GENTDEFR( zipsc )

