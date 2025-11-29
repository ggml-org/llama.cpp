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


void bli_zdotaxpyv_template_noopt
     (
       conj_t             conjxt,
       conj_t             conjx,
       conj_t             conjy,
       dim_t              n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       dcomplex* restrict rho,
       dcomplex* restrict z, inc_t incz,
       cntx_t*            cntx
     )
{
/*
  Template dotaxpyv kernel implementation

  This function contains a template implementation for a double-precision
  complex kernel, coded in C, which can serve as the starting point for one
  to write an optimized kernel on an arbitrary architecture. (We show a
  template implementation for only double-precision complex because the
  templates for the other three floating-point types would be similar, with
  the real instantiations being noticeably simpler due to the disappearance
  of conjugation in the real domain.)

  This kernel fuses a dotv and axpyv operation:

    rho := conjxt( x^T ) * conjy( y )
    z   := z + alpha * conjx( x )

  where x, y, and z are vectors of length n and alpha1 and alpha2 are scalars.

  Parameters:

  - conjxt: Compute with conjugated values of x^T?
  - conjx:  Compute with conjugated values of x?
  - conjy:  Compute with conjugated values of y?
  - n:      The number of elements in vectors x, y, and z.
  - alpha:  The address of the scalar to be applied to x.
  - x:      The address of vector x.
  - incx:   The vector increment of x. incx should be unit unless the
            implementation makes special accomodation for non-unit values.
  - y:      The address of vector y.
  - incy:   The vector increment of y. incy should be unit unless the
            implementation makes special accomodation for non-unit values.
  - rho:    The address of the output scalar of the dotv subproblem.
  - z:      The address of vector z.
  - incz:   The vector increment of z. incz should be unit unless the
            implementation makes special accomodation for non-unit values.

  This template code calls the reference implementation if any of the
  following conditions are true:

  - Any of the strides incx, incy, or incz is non-unit.
  - Vectors x, y, and z are unaligned with different offsets.

  If the vectors are aligned, or unaligned by the same offset, then optimized
  code can be used for the bulk of the computation. This template shows how
  the front-edge case can be handled so that the remaining computation is
  aligned. (This template guarantees alignment in the main loops to be
  BLIS_SIMD_ALIGN_SIZE.)

  Here are a few additional things to consider:

  - While four combinations of possible values of conjx and conjy exist, we
    implement only conjugation on x explicitly; we induce the other two cases
    by toggling the effective conjugation on x and then conjugating the dot
    product result.
  - Because conjugation disappears in the real domain, real instances of
    this kernel can safely ignore the values of any conjugation parameters,
    thereby simplifying the implementation.

  For more info, please refer to the BLIS website and/or contact the
  blis-devel mailing list.

  -FGVZ
*/
	const dim_t n_elem_per_reg  = 1;
	const dim_t n_iter_unroll   = 1;

	const dim_t n_elem_per_iter = n_elem_per_reg * n_iter_unroll;
	const siz_t type_size       = sizeof( *x );

	dcomplex*   xp;
	dcomplex*   yp;
	dcomplex*   zp;
	dcomplex    dotxy;

	bool        use_ref         = FALSE;

	dim_t       n_pre           = 0;
	dim_t       n_iter;
	dim_t       n_left;

	dim_t       off_x, off_y, off_z;
	dim_t       i;

	conj_t      conjxt_use;


	// If the vector lengths are zero, set rho to zero and return.
	if ( bli_zero_dim1( n ) )
	{
		bli_tset0s( z, *rho );
		return;
	}

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( bli_has_nonunit_inc3( incx, incy, incz ) )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( x, BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( y, BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( z, BLIS_SIMD_ALIGN_SIZE ) )
	{
		use_ref = TRUE;

		// If x, y, and z are unaligned by the same offset, then we can
		// still use an implementation that depends on alignment for most
		// of the operation.
		off_x = bli_offset_from_alignment( x, BLIS_SIMD_ALIGN_SIZE );
		off_y = bli_offset_from_alignment( y, BLIS_SIMD_ALIGN_SIZE );
		off_z = bli_offset_from_alignment( z, BLIS_SIMD_ALIGN_SIZE );

		if ( off_x == off_y && off_x == off_z )
		{
			use_ref = FALSE;
			n_pre   = off_x / type_size;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		zdotaxpyv_ft f = bli_zdotaxpyv_template_ref;

		f
		(
		  conjxt,
		  conjx,
		  conjy,
		  n,
		  alpha,
		  x, incx,
		  y, incy,
		  rho,
		  z, incz,
		  cntx
		);
        return;
	}


	// Compute the number of unrolled and leftover (edge) iterations.
	n_iter = ( n - n_pre ) / n_elem_per_iter;
	n_left = ( n - n_pre ) % n_elem_per_iter;


	// Initialize pointers into x, y, and z.
	xp = x;
	yp = y;
	zp = z;


	// Initialize accumulator to zero.
	bli_tset0s( z, dotxy );


	conjxt_use = conjxt;

	// If y must be conjugated, we compute the result indirectly by first
	// toggling the effective conjugation of xt and then conjugating the
	// resulting dot product.
	if ( bli_is_conj( conjy ) )
		bli_toggle_conj( &conjxt_use );


	// Iterate over elements of x, y, and z to compute:
	//   r = conjxt( x^T ) * conjy( y );
	//   z += alpha * conjx( x );
	if ( bli_is_noconj( conjx ) && bli_is_noconj( conjxt_use ) )
	{
		// Compute front edge cases if x, y, and z were unaligned.
		for ( i = 0; i < n_pre; ++i )
		{
			bli_tdots( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpys( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// alpha should be loaded once prior to the n_iter loop, dotxy
		// should be and kept in registers, and each element of x should be
		// loaded only once each. The addresses xp, yp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < n_iter; ++i )
		{
			bli_tdots( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpys( z,z,z,z, *alpha, *xp, *zp );

			xp += n_elem_per_iter;
			yp += n_elem_per_iter;
			zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < n_left; ++i )
		{
			bli_tdots( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpys( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}
	}
	else if ( bli_is_noconj( conjx ) && bli_is_conj( conjxt_use ) )
	{
		// Compute front edge cases if x, y, and z were unaligned.
		for ( i = 0; i < n_pre; ++i )
		{
			bli_tdotjs( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpys( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// alpha should be loaded once prior to the n_iter loop, dotxy
		// should be and kept in registers, and each element of x should be
		// loaded only once each. The addresses xp, yp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < n_iter; ++i )
		{
			bli_tdotjs( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpys( z,z,z,z, *alpha, *xp, *zp );

			xp += n_elem_per_iter;
			yp += n_elem_per_iter;
			zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < n_left; ++i )
		{
			bli_tdotjs( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpys( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}
	}
	else if ( bli_is_conj( conjx ) && bli_is_noconj( conjxt_use ) )
	{
		// Compute front edge cases if x, y, and z were unaligned.
		for ( i = 0; i < n_pre; ++i )
		{
			bli_tdots( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// alpha should be loaded once prior to the n_iter loop, dotxy
		// should be and kept in registers, and each element of x should be
		// loaded only once each. The addresses xp, yp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < n_iter; ++i )
		{
			bli_tdots( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *zp );

			xp += n_elem_per_iter;
			yp += n_elem_per_iter;
			zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < n_left; ++i )
		{
			bli_tdots( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}
	}
	else // if ( bli_is_conj( conjx ) && bli_is_conj( conjxt_use ) )
	{
		// Compute front edge cases if x, y, and z were unaligned.
		for ( i = 0; i < n_pre; ++i )
		{
			bli_tdotjs( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// alpha should be loaded once prior to the n_iter loop, dotxy
		// should be and kept in registers, and each element of x should be
		// loaded only once each. The addresses xp, yp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < n_iter; ++i )
		{
			bli_tdotjs( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *zp );

			xp += n_elem_per_iter;
			yp += n_elem_per_iter;
			zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < n_left; ++i )
		{
			bli_tdotjs( z,z,z,z, *xp, *yp, dotxy );
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *zp );

			xp += 1; yp += 1; zp += 1;
		}
	}

	// If conjugation on y was requested, we induce it by conjugating
	// the contents of rho.
	if ( bli_is_conj( conjy ) )
		bli_tconjs( z, dotxy );

	bli_tcopys( z,z, dotxy, *rho );
}

