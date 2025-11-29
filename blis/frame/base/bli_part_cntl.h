/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

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


struct part_cntl_s
{
	cntl_t cntl; // cntl field must be present and come first.
	num_t  b_dt;
	dim_t  b_alg;
	dim_t  b_max;
	dim_t  b_scale;
	dim_t  b_mult;
	dim_t  b_mult_scale;
	dir_t  direct;
	bool   use_weighted;
};
typedef struct part_cntl_s part_cntl_t;

// -----------------------------------------------------------------------------

BLIS_INLINE dim_t bli_part_cntl_blksz_alg( const cntl_t* cntl )
{
	return ( ( const part_cntl_t* )cntl )->b_alg;
}

BLIS_INLINE dim_t bli_part_cntl_blksz_max( const cntl_t* cntl )
{
	return ( ( const part_cntl_t* )cntl )->b_max;
}

BLIS_INLINE dim_t bli_part_cntl_blksz_mult( const cntl_t* cntl )
{
	return ( ( const part_cntl_t* )cntl )->b_mult;
}

BLIS_INLINE dir_t bli_part_cntl_direct( const cntl_t* cntl )
{
	return ( ( const part_cntl_t* )cntl )->direct;
}

BLIS_INLINE bool bli_part_cntl_use_weighted( const cntl_t* cntl )
{
	return ( ( const part_cntl_t* )cntl )->use_weighted;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_part_cntl_set_blksz( const blksz_t* blksz, cntl_t* cntl_ )
{
	part_cntl_t* cntl = ( part_cntl_t* )cntl_;
	num_t dt = cntl->b_dt;
	cntl->b_alg = bli_blksz_get_def( dt, blksz ) / cntl->b_scale;
	cntl->b_max = bli_blksz_get_max( dt, blksz ) / cntl->b_scale;
}

BLIS_INLINE void bli_part_cntl_set_blksz_mult( const blksz_t* blksz, cntl_t* cntl_ )
{
	part_cntl_t* cntl = ( part_cntl_t* )cntl_;
	num_t dt = cntl->b_dt;
	cntl->b_mult = bli_blksz_get_def( dt, blksz ) / cntl->b_mult_scale;
}

BLIS_INLINE void bli_part_cntl_set_direct( dir_t direct, cntl_t* cntl )
{
	( ( part_cntl_t* )cntl )->direct = direct;
}

BLIS_INLINE void bli_part_cntl_set_use_weighted( bool use_weighted, cntl_t* cntl )
{
	( ( part_cntl_t* )cntl )->use_weighted = use_weighted;
}

BLIS_INLINE void bli_part_cntl_align_blksz_to_mult( dim_t mult, bool round_up, cntl_t* cntl_ )
{
	part_cntl_t* cntl = ( part_cntl_t* )cntl_;
	cntl->b_alg = bli_align_dim_to_mult( cntl->b_alg, mult, round_up );
	cntl->b_max = bli_align_dim_to_mult( cntl->b_max, mult, round_up );
}

BLIS_INLINE void bli_part_cntl_align_blksz( bool round_up, cntl_t* cntl )
{
	bli_part_cntl_align_blksz_to_mult( ( ( part_cntl_t* )cntl )->b_mult, round_up, cntl );
}

// -----------------------------------------------------------------------------

void bli_part_cntl_init_node
     (
       void_fp      var_func,
       num_t        b_dt,
       dim_t        b_alg,
       dim_t        b_max,
       dim_t        b_scale,
       dim_t        b_mult,
       dim_t        b_mult_scale,
       dir_t        direct,
       bool         use_weighted,
       part_cntl_t* cntl
     );

