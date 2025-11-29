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

#ifndef BLIS_ARCH_CONFIG_PRE_H
#define BLIS_ARCH_CONFIG_PRE_H


// -- Naming-related kernel definitions ----------------------------------------

// The default suffix appended to reference kernels.
#define BLIS_REF_SUFFIX  _ref

// A suffix used for labeling certain induced method aware functions.
#define BLIS_IND_SUFFIX  _ind

// Add an underscore to the BLIS kernel set string, if it was defined.
#ifdef  BLIS_CNAME
#define BLIS_CNAME_INFIX  PASTECH(_,BLIS_CNAME)
#endif

// Add an underscore to the BLIS kernel set string, if it was defined.
#ifdef  BLIS_CNAME_UPPER
#define BLIS_CNAME_UPPER_INFIX  PASTECH(_,BLIS_CNAME_UPPER)
#endif

// Add an underscore to the plugin name, if it was defined.
#ifdef  BLIS_PNAME
#define BLIS_PNAME_INFIX  PASTECH(_,BLIS_PNAME)
#endif

// Combine the CNAME and _ref for convenience to the code that defines
// reference kernels.
//#define BLIS_CNAME_REF_SUFFIX  PASTECH(_,BLIS_CNAME,BLIS_REF_SUFFIX)

// -- Prototype-generating macro definitions -----------------------------------

// Prototype-generating macro for bli_cntx_init_<arch>*() functions.
#define CNTX_INIT_PROTS( archname ) \
\
void PASTEMAC(cntx_init_,archname) \
     ( \
       cntx_t* cntx \
     ); \
void PASTEMAC(cntx_init_,archname,BLIS_REF_SUFFIX) \
     ( \
       cntx_t* cntx \
     ); \
void PASTEMAC(cntx_init_,archname,BLIS_IND_SUFFIX) \
     ( \
       ind_t   method, \
       cntx_t* cntx \
     );


#endif

