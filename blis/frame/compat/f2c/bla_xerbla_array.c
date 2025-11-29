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

#include "blis.h"

#ifdef BLIS_ENABLE_BLAS

#define MAX_NUM_CHARS 32

#ifndef BLIS_DISABLE_XERBLA_ARRAY
int PASTEF77(xerbla_array)(const bla_character *srname_array, const bla_integer srname_len, const bla_integer *info)
{
	int  i;
#if 1
	//                                  01234567890123456789012345678901
	char srname[ MAX_NUM_CHARS + 1 ] = "                                ";
#else
	char srname[ MAX_NUM_CHARS + 1 ];

	// Initialize srname to contain blank characters.
	for ( i = 0; i < MAX_NUM_CHARS; ++i ) srname[i] = ' ';
#endif

	// Compute the number of chars to copy as the minimum of the length of
	// srname_array and MAX_NUM_CHARS.
	const int n_copy = bli_min( srname_len, MAX_NUM_CHARS );

	// Copy over each element of srname_array.
	for ( i = 0; i < n_copy; ++i )
	{
		srname[i] = srname_array[i];
	}

	// NULL terminate.
	srname[i] = '\0';

	// Call xerbla_().
	PASTEF77(xerbla)( srname, info, ( ftnlen )srname_len );

	return 0;
}

#endif
#endif

