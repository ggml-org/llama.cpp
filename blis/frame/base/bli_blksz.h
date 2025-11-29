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

// blksz_t query

BLIS_INLINE dim_t bli_blksz_get_def
     (
             num_t    dt,
       const blksz_t* b
     )
{
	return b->v[ dt ];
}

BLIS_INLINE dim_t bli_blksz_get_max
     (
             num_t    dt,
       const blksz_t* b
     )
{
	return b->e[ dt ];
}


// blksz_t modification

BLIS_INLINE void bli_blksz_set_def
     (
       dim_t    val,
       num_t    dt,
       blksz_t* b
     )
{
	b->v[ dt ] = val;
}

BLIS_INLINE void bli_blksz_set_max
     (
       dim_t    val,
       num_t    dt,
       blksz_t* b
     )
{
	b->e[ dt ] = val;
}

BLIS_INLINE void bli_blksz_copy
     (
       const blksz_t* b_src,
             blksz_t* b_dst
     )
{
	*b_dst = *b_src;
}

BLIS_INLINE void bli_blksz_copy_if_nonneg
     (
       const blksz_t* b_src,
             blksz_t* b_dst
     )
{
	// Copy the blocksize values over to b_dst one-by-one. Note that we
	// only copy valuse that are zero or positive (and skip copying any
	// values that are negative).

	const dim_t v_s = bli_blksz_get_def( BLIS_FLOAT,    b_src );
	const dim_t v_d = bli_blksz_get_def( BLIS_DOUBLE,   b_src );
	const dim_t v_c = bli_blksz_get_def( BLIS_SCOMPLEX, b_src );
	const dim_t v_z = bli_blksz_get_def( BLIS_DCOMPLEX, b_src );

	const dim_t e_s = bli_blksz_get_max( BLIS_FLOAT,    b_src );
	const dim_t e_d = bli_blksz_get_max( BLIS_DOUBLE,   b_src );
	const dim_t e_c = bli_blksz_get_max( BLIS_SCOMPLEX, b_src );
	const dim_t e_z = bli_blksz_get_max( BLIS_DCOMPLEX, b_src );

	if ( v_s >= 0 ) bli_blksz_set_def( v_s, BLIS_FLOAT,    b_dst );
	if ( v_d >= 0 ) bli_blksz_set_def( v_d, BLIS_DOUBLE,   b_dst );
	if ( v_c >= 0 ) bli_blksz_set_def( v_c, BLIS_SCOMPLEX, b_dst );
	if ( v_z >= 0 ) bli_blksz_set_def( v_z, BLIS_DCOMPLEX, b_dst );

	if ( e_s >= 0 ) bli_blksz_set_max( e_s, BLIS_FLOAT,    b_dst );
	if ( e_d >= 0 ) bli_blksz_set_max( e_d, BLIS_DOUBLE,   b_dst );
	if ( e_c >= 0 ) bli_blksz_set_max( e_c, BLIS_SCOMPLEX, b_dst );
	if ( e_z >= 0 ) bli_blksz_set_max( e_z, BLIS_DCOMPLEX, b_dst );
}

BLIS_INLINE void bli_blksz_copy_def_dt
     (
       num_t dt_src, const blksz_t* b_src,
       num_t dt_dst,       blksz_t* b_dst
     )
{
	const dim_t val = bli_blksz_get_def( dt_src, b_src );

	bli_blksz_set_def( val, dt_dst, b_dst );
}

BLIS_INLINE void bli_blksz_copy_max_dt
     (
       num_t dt_src, const blksz_t* b_src,
       num_t dt_dst,       blksz_t* b_dst
     )
{
	const dim_t val = bli_blksz_get_max( dt_src, b_src );

	bli_blksz_set_max( val, dt_dst, b_dst );
}

BLIS_INLINE void bli_blksz_copy_dt
     (
       num_t dt_src, const blksz_t* b_src,
       num_t dt_dst,       blksz_t* b_dst
     )
{
	bli_blksz_copy_def_dt( dt_src, b_src, dt_dst, b_dst );
	bli_blksz_copy_max_dt( dt_src, b_src, dt_dst, b_dst );
}

BLIS_INLINE void bli_blksz_scale_def
     (
       dim_t    num,
       dim_t    den,
       num_t    dt,
       blksz_t* b
     )
{
	const dim_t val = bli_blksz_get_def( dt, b );

	bli_blksz_set_def( ( val * num ) / den, dt, b );
}

BLIS_INLINE void bli_blksz_scale_max
     (
       dim_t    num,
       dim_t    den,
       num_t    dt,
       blksz_t* b
     )
{
	const dim_t val = bli_blksz_get_max( dt, b );

	bli_blksz_set_max( ( val * num ) / den, dt, b );
}

BLIS_INLINE void bli_blksz_scale_def_max
     (
       dim_t    num,
       dim_t    den,
       num_t    dt,
       blksz_t* b
     )
{
	bli_blksz_scale_def( num, den, dt, b );
	bli_blksz_scale_max( num, den, dt, b );
}

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS blksz_t* bli_blksz_create_ed
     (
       dim_t b_s, dim_t be_s,
       dim_t b_d, dim_t be_d,
       dim_t b_c, dim_t be_c,
       dim_t b_z, dim_t be_z
     );

BLIS_EXPORT_BLIS blksz_t* bli_blksz_create
     (
       dim_t b_s,  dim_t b_d,  dim_t b_c,  dim_t b_z,
       dim_t be_s, dim_t be_d, dim_t be_c, dim_t be_z
     );

BLIS_EXPORT_BLIS void bli_blksz_init_ed
     (
       blksz_t* b,
       dim_t    b_s, dim_t be_s,
       dim_t    b_d, dim_t be_d,
       dim_t    b_c, dim_t be_c,
       dim_t    b_z, dim_t be_z
     );

BLIS_EXPORT_BLIS void bli_blksz_init
     (
       blksz_t* b,
       dim_t b_s,  dim_t b_d,  dim_t b_c,  dim_t b_z,
       dim_t be_s, dim_t be_d, dim_t be_c, dim_t be_z
     );

BLIS_EXPORT_BLIS void bli_blksz_init_easy
     (
       blksz_t* b,
       dim_t b_s,  dim_t b_d,  dim_t b_c,  dim_t b_z
     );

BLIS_EXPORT_BLIS void bli_blksz_free
     (
       blksz_t* b
     );

// -----------------------------------------------------------------------------

#if 0
BLIS_EXPORT_BLIS void bli_blksz_reduce_dt_to
     (
       num_t dt_bm, blksz_t* bmult,
       num_t dt_bs, blksz_t* blksz
     );
#endif

void bli_blksz_reduce_def_to
     (
       num_t dt_bm, blksz_t* bmult,
       num_t dt_bs, blksz_t* blksz
     );

void bli_blksz_reduce_max_to
     (
       num_t dt_bm, blksz_t* bmult,
       num_t dt_bs, blksz_t* blksz
     );
// -----------------------------------------------------------------------------

dim_t bli_determine_blocksize
     (
       dir_t direct,
       dim_t i,
       dim_t dim,
       dim_t b_alg,
       dim_t b_max
     );

