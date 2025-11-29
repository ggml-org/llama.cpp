/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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


/*
// -- Control tree node definition --

struct cntl_s
{
	l3_var_oft var_func;
	struct
	{
		dim_t          ways;
		struct cntl_s* sub_node;
	} sub_nodes[ BLIS_MAX_SUB_NODES ];
};
typedef struct cntl_s cntl_t;
*/


// -- Control tree prototypes --

BLIS_EXPORT_BLIS void bli_cntl_init_node
     (
       void_fp var_func,
       cntl_t* cntl
     );

BLIS_EXPORT_BLIS void bli_cntl_attach_sub_node
     (
       dim_t   ways,
       cntl_t* sub_node,
       cntl_t* cntl
     );

BLIS_EXPORT_BLIS void bli_cntl_insert_sub_node
     (
       dim_t   ways,
       cntl_t* sub_node,
       cntl_t* child_node,
       cntl_t* cntl
     );

BLIS_EXPORT_BLIS void bli_cntl_clear_node
     (
       cntl_t* cntl
     );

// -----------------------------------------------------------------------------

// cntl_t query (fields only)

BLIS_INLINE void_fp bli_cntl_var_func( const cntl_t* cntl )
{
	return cntl->var_func;
}

BLIS_INLINE dim_t bli_cntl_ways( dim_t which, const cntl_t* cntl )
{
	return cntl->sub_nodes[ which ].ways;
}

BLIS_INLINE cntl_t* bli_cntl_sub_node( dim_t which, const cntl_t* cntl )
{
	return cntl->sub_nodes[ which ].sub_node;
}

// cntl_t query (complex)

BLIS_INLINE bool bli_cntl_is_leaf( const cntl_t* cntl )
{
	return ( bool )
	       ( bli_cntl_sub_node( 0, cntl ) == NULL );
}

// cntl_t modification

BLIS_INLINE void bli_cntl_set_var_func( void_fp var_func, cntl_t* cntl )
{
	cntl->var_func = var_func;
}

BLIS_INLINE void bli_cntl_set_ways( dim_t which, dim_t ways, cntl_t* cntl )
{
	cntl->sub_nodes[ which ].ways = ways;
}

BLIS_INLINE void bli_cntl_set_sub_node( dim_t which, cntl_t* sub_node, cntl_t* cntl )
{
	cntl->sub_nodes[ which ].sub_node = sub_node;
}

