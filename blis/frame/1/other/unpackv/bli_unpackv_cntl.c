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

unpackv_t* unpackv_cntl = NULL;

void bli_unpackv_cntl_init()
{
	unpackv_cntl = bli_unpackv_cntl_obj_create( BLIS_UNBLOCKED,
	                                            BLIS_VARIANT1 );
}

void bli_unpackv_cntl_finalize()
{
	bli_cntl_free_node( unpackv_cntl );
}

unpackv_t* bli_unpackv_cntl_obj_create( impl_t     impl_type,
                                        varnum_t   var_num )
{
	unpackv_t* cntl;

	cntl = ( unpackv_t* ) bli_malloc_intl( sizeof(unpackv_t) );

	cntl->impl_type = impl_type;
	cntl->var_num   = var_num;

	return cntl;
}

void bli_unpackv_cntl_obj_init( unpackv_t* cntl,
                                impl_t     impl_type,
                                varnum_t   var_num )
{
	cntl->impl_type = impl_type;
	cntl->var_num   = var_num;
}

