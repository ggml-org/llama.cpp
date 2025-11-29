/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2024, Southern Methodist University

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

void bli_cntl_init_node
     (
       void_fp var_func,
       cntl_t* cntl
     )
{
	bli_cntl_set_var_func( var_func, cntl );
	for ( dim_t i = 0; i < BLIS_MAX_SUB_NODES; i++ )
	{
		bli_cntl_set_ways( i, 0, cntl );
		bli_cntl_set_sub_node( i, NULL, cntl );
	}
}

void bli_cntl_attach_sub_node
     (
       dim_t   ways,
       cntl_t* sub_node,
       cntl_t* cntl
     )
{
	dim_t next = 0;
	for ( ; next < BLIS_MAX_SUB_NODES; next++ )
	{
		if ( bli_cntl_sub_node( next, cntl ) == NULL )
			break;
	}

	if ( next == BLIS_MAX_SUB_NODES )
		bli_abort();

	bli_cntl_set_ways( next, ways, cntl );
	bli_cntl_set_sub_node( next, sub_node, cntl );
}

void bli_cntl_insert_sub_node
     (
       dim_t   ways,
       cntl_t* sub_node,
       cntl_t* child_node,
       cntl_t* cntl
     )
{
	dim_t next = 0;
	for ( ; next < BLIS_MAX_SUB_NODES; next++ )
	{
		if ( bli_cntl_sub_node( next, cntl ) == child_node )
			break;
	}

	if ( next == BLIS_MAX_SUB_NODES )
		bli_abort();

	bli_cntl_attach_sub_node( bli_cntl_ways( next, cntl ), child_node, sub_node );
	bli_cntl_set_ways( next, ways, cntl );
	bli_cntl_set_sub_node( next, sub_node, cntl );
}

void bli_cntl_clear_node
     (
       cntl_t* cntl
     )
{
	bli_cntl_init_node( NULL, cntl );
}

