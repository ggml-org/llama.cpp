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

#ifndef BLIS_AUXINFO_MACRO_DEFS_H
#define BLIS_AUXINFO_MACRO_DEFS_H


// auxinfo_t field query

BLIS_INLINE pack_t bli_auxinfo_schema_a( const auxinfo_t* ai )
{
	return ai->schema_a;
}
BLIS_INLINE pack_t bli_auxinfo_schema_b( const auxinfo_t* ai )
{
	return ai->schema_b;
}

BLIS_INLINE const void* bli_auxinfo_next_a( const auxinfo_t* ai )
{
	return ai->a_next;
}
BLIS_INLINE const void* bli_auxinfo_next_b( const auxinfo_t* ai )
{
	return ai->b_next;
}

BLIS_INLINE inc_t bli_auxinfo_is_a( const auxinfo_t* ai )
{
	return ai->is_a;
}
BLIS_INLINE inc_t bli_auxinfo_is_b( const auxinfo_t* ai )
{
	return ai->is_b;
}

BLIS_INLINE inc_t bli_auxinfo_ps_a( const auxinfo_t* ai )
{
	return ai->ps_a;
}
BLIS_INLINE inc_t bli_auxinfo_ps_b( const auxinfo_t* ai )
{
	return ai->ps_b;
}

BLIS_INLINE inc_t bli_auxinfo_off_m( const auxinfo_t* ai )
{
	return ai->off_m;
}
BLIS_INLINE inc_t bli_auxinfo_off_n( const auxinfo_t* ai )
{
	return ai->off_n;
}

BLIS_INLINE void_fp bli_auxinfo_ukr( const auxinfo_t* ai )
{
	return ai->ukr;
}
BLIS_INLINE const void* bli_auxinfo_params( const auxinfo_t* ai )
{
	return ai->params;
}


// auxinfo_t field modification

BLIS_INLINE void bli_auxinfo_set_schema_a( pack_t schema, auxinfo_t* ai )
{
	ai->schema_a = schema;
}
BLIS_INLINE void bli_auxinfo_set_schema_b( pack_t schema, auxinfo_t* ai )
{
	ai->schema_b = schema;
}

BLIS_INLINE void bli_auxinfo_set_next_a( const void* p, auxinfo_t* ai )
{
	ai->a_next = p;
}
BLIS_INLINE void bli_auxinfo_set_next_b( const void* p, auxinfo_t* ai )
{
	ai->b_next = p;
}
BLIS_INLINE void bli_auxinfo_set_next_ab( const void* ap, const void* bp, auxinfo_t* ai )
{
	ai->a_next = ap;
	ai->b_next = bp;
}

BLIS_INLINE void bli_auxinfo_set_is_a( inc_t is, auxinfo_t* ai )
{
	ai->is_a = is;
}
BLIS_INLINE void bli_auxinfo_set_is_b( inc_t is, auxinfo_t* ai )
{
	ai->is_b = is;
}

BLIS_INLINE void bli_auxinfo_set_ps_a( inc_t ps, auxinfo_t* ai )
{
	ai->ps_a = ps;
}
BLIS_INLINE void bli_auxinfo_set_ps_b( inc_t ps, auxinfo_t* ai )
{
	ai->ps_b = ps;
}

BLIS_INLINE void bli_auxinfo_set_off_m( dim_t off, auxinfo_t* ai )
{
	ai->off_m = off;
}
BLIS_INLINE void bli_auxinfo_set_off_n( dim_t off, auxinfo_t* ai )
{
	ai->off_n = off;
}

BLIS_INLINE void bli_auxinfo_set_ukr( void_fp ukr, auxinfo_t* ai )
{
	ai->ukr = ukr;
}
BLIS_INLINE void bli_auxinfo_set_params( const void* params, auxinfo_t* ai )
{
	ai->params = params;
}

#endif

