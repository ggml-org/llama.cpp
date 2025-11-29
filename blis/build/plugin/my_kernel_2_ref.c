/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Southern Methodist University

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

#include @PLUGIN_HEADER@

#ifndef MY_KERNEL_2_ROW_MAJOR
#define MY_KERNEL_2_ROW_MAJOR 0
#endif

#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
       int    m, \
       int    n, \
       ctype* a  \
     ) \
{ \
	if ( bli_zero_dim1( m ) || bli_zero_dim1( n ) ) return; \
\
	if ( MY_KERNEL_2_ROW_MAJOR ) \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				bli_tseti0s( ch, a[ i*n + j ] ); \
			} \
		} \
	} \
	else \
	{ \
		for ( dim_t i = 0; i < m; ++i ) \
		{ \
			for ( dim_t j = 0; j < n; ++j ) \
			{ \
				bli_tseti0s( ch, a[ i + j*m ] ); \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC( my_kernel_2, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

