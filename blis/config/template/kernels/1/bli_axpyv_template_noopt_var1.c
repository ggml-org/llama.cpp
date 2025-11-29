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


void bli_zaxpyv_template_noopt
     (
       conj_t             conjx,
       dim_t              n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t*            cntx
     )
{
/*
  Template axpyv kernel implementation

  This function contains a template implementation for a double-precision
  complex kernel, coded in C, which can serve as the starting point for one
  to write an optimized kernel on an arbitrary architecture. (We show a
  template implementation for only double-precision complex because the
  templates for the other three floating-point types would be similar, with
  the real instantiations being noticeably simpler due to the disappearance
  of conjugation in the real domain.)

  This kernel performs a vector scale and accumulate (axpy) operation:

    y := y + alpha * conjx( x )

  where x and y are vectors of length n and alpha is a scalar.

  Parameters:

  - conjx:  Compute with conjugated values of x?
  - n:      The number of elements in vectors x and y.
  - alpha:  The address of a scalar.
  - x:      The address of vector x.
  - incx:   The vector increment of x. incx should be unit unless the
            implementation makes special accomodation for non-unit values.
  - y:      The address of vector y.
  - incy:   The vector increment of y. incy should be unit unless the
            implementation makes special accomodation for non-unit values.

  This template code calls the reference implementation if any of the
  following conditions are true:

  - Either of the strides incx or incy is non-unit.
  - Vectors x and y are unaligned with different offsets.

  If the vectors are aligned, or unaligned by the same offset, then optimized
  code can be used for the bulk of the computation. This template shows how
  the front-edge case can be handled so that the remaining computation is
  aligned. (This template guarantees alignment to be BLIS_SIMD_ALIGN_SIZE.)

  Additional things to consider:

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

	bool        use_ref         = FALSE;

	dim_t       n_pre           = 0;
	dim_t       n_iter;
	dim_t       n_left;

	dim_t       off_x, off_y;
	dim_t       i;


	if ( bli_zero_dim1( n ) ) return;

	if ( bli_teq0s( z, *alpha ) ) return;


	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( bli_has_nonunit_inc2( incx, incy ) )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( x, BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( y, BLIS_SIMD_ALIGN_SIZE ) )
	{
		use_ref = TRUE;

		// If a, the second column of a, and y are unaligned by the same
		// offset, then we can still use an implementation that depends on
		// alignment for most of the operation.
		off_x  = bli_offset_from_alignment( x, BLIS_SIMD_ALIGN_SIZE );
		off_y  = bli_offset_from_alignment( y, BLIS_SIMD_ALIGN_SIZE );

		if ( off_x == off_y )
		{
			use_ref = FALSE;
			n_pre   = off_x / type_size;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		zaxpyv_ft f = bli_zaxpyv_template_ref;

		f
		(
		  conjx,
		  n,
		  alpha,
		  x, incx,
		  y, incy,
		  cntx
		);
        return;
	}


	// Compute the number of unrolled and leftover (edge) iterations.
	n_iter = ( n - n_pre ) / n_elem_per_iter;
	n_left = ( n - n_pre ) % n_elem_per_iter;


	// Initialize pointers into x and y.
	xp = x;
	yp = y;


	// Iterate over elements of x and y to compute:
	//  y += alpha * conjx( x );
	if ( bli_is_noconj( conjx ) )
	{
		// Compute front edge cases if x and y were unaligned.
		for ( i = 0; i < n_pre; ++i )
		{
			bli_taxpys( z,z,z,z, *alpha, *xp, *yp );

			xp += 1; yp += 1;
		}

		// The bulk of the operation is executed here. The addresses xp and
		// yp are guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < n_iter; ++i )
		{
			bli_taxpys( z,z,z,z, *alpha, *xp, *yp );

			xp += n_elem_per_iter;
			yp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < n_left; ++i )
		{
			bli_taxpys( z,z,z,z, *alpha, *xp, *yp );

			xp += 1; yp += 1;
		}
	}
	else // if ( bli_is_conj( conjx ) )
	{
		// Compute front edge cases if x and y were unaligned.
		for ( i = 0; i < n_pre; ++i )
		{
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *yp );

			xp += 1; yp += 1;
		}

		// The bulk of the operation is executed here. The addresses xp and
		// yp are guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < n_iter; ++i )
		{
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *yp );

			xp += n_elem_per_iter;
			yp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < n_left; ++i )
		{
			bli_taxpyjs( z,z,z,z, *alpha, *xp, *yp );

			xp += 1; yp += 1;
		}
	}
}

