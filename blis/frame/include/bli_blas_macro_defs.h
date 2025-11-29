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

#ifndef BLIS_BLAS_MACRO_DEFS_H
#define BLIS_BLAS_MACRO_DEFS_H

// -- Various Fortran compatibility macros --

// Macro to treat negative dimensions as zero.

#define bli_convert_blas_dim1( n_blas, n_blis )\
{ \
	if ( n_blas < 0 ) n_blis = ( dim_t )0; \
	else              n_blis = ( dim_t )n_blas; \
}

// Macro to flip signs of increments if input increments are negative.

#define bli_convert_blas_incv( n, x_blas, incx_blas, \
                                  x_blis, incx_blis ) \
{ \
	if ( incx_blas < 0 ) \
	{ \
		/* The semantics of negative stride in BLAS are that the vector
		   operand be traversed in reverse order. (Another way to think
		   of this is that negative strides effectively reverse the order
		   of the vector, but without any explicit data movements.) This
		   is also how BLIS interprets negative strides. The differences
		   is that with BLAS, the caller *always* passes in the 0th (i.e.,
		   top-most or left-most) element of the vector, even when the
		   stride is negative. By contrast, in BLIS, negative strides are
		   used *relative* to the vector address as it is given. Thus, in
		   BLIS, if this backwards traversal is desired, the caller *must*
		   pass in the address to the (n-1)th (i.e., the bottom-most or
		   right-most) element along with a negative stride. */ \
		x_blis    = (x_blas) + (n-1)*(-incx_blas); \
		incx_blis = ( inc_t )(incx_blas); \
	} \
	else \
	{ \
		x_blis    = (x_blas); \
		incx_blis = ( inc_t )(incx_blas); \
	} \
}



#endif

