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
// Prototype function pointer query interface.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) \
PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( num_t dt );

GENPROT( gemv )
GENPROT( ger )
GENPROT( hemv )
GENPROT( symv )
GENPROT( her )
GENPROT( syr )
GENPROT( her2 )
GENPROT( syr2 )
GENPROT( trmv )
GENPROT( trsv )

//
// Prototype function pointer query interfaces for level-2 implementations.
//

#undef  GENPROT
#define GENPROT( opname, varname ) \
\
PASTECH(opname,_unb,_vft) \
PASTEMAC(varname,_qfp)( num_t dt );

GENPROT( gemv, gemv_unb_var1 )
GENPROT( gemv, gemv_unb_var2 )
GENPROT( gemv, gemv_unf_var1 )
GENPROT( gemv, gemv_unf_var2 )

GENPROT( ger, ger_unb_var1 )
GENPROT( ger, ger_unb_var2 )

GENPROT( hemv, hemv_unb_var1 )
GENPROT( hemv, hemv_unb_var2 )
GENPROT( hemv, hemv_unb_var3 )
GENPROT( hemv, hemv_unb_var4 )
GENPROT( hemv, hemv_unf_var1 )
GENPROT( hemv, hemv_unf_var3 )
GENPROT( hemv, hemv_unf_var1a )
GENPROT( hemv, hemv_unf_var3a )

GENPROT( her, her_unb_var1 )
GENPROT( her, her_unb_var2 )

GENPROT( her2, her2_unb_var1 )
GENPROT( her2, her2_unb_var2 )
GENPROT( her2, her2_unb_var3 )
GENPROT( her2, her2_unb_var4 )
GENPROT( her2, her2_unf_var1 )
GENPROT( her2, her2_unf_var4 )

GENPROT( trmv, trmv_unb_var1 )
GENPROT( trmv, trmv_unb_var2 )
GENPROT( trmv, trmv_unf_var1 )
GENPROT( trmv, trmv_unf_var2 )

GENPROT( trsv, trsv_unb_var1 )
GENPROT( trsv, trsv_unb_var2 )
GENPROT( trsv, trsv_unf_var1 )
GENPROT( trsv, trsv_unf_var2 )

