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


void bli_zaxpyf_template_noopt
     (
       conj_t             conja,
       conj_t             conjx,
       dim_t              m,
       dim_t              b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t*            cntx
     )
{
/*
  Template axpyf kernel implementation

  This function contains a template implementation for a double-precision
  complex kernel, coded in C, which can serve as the starting point for one
  to write an optimized kernel on an arbitrary architecture. (We show a
  template implementation for only double-precision complex because the
  templates for the other three floating-point types would be similar, with
  the real instantiations being noticeably simpler due to the disappearance
  of conjugation in the real domain.)

  This kernel performs the following gemv-like operation:

    y := y + alpha * conja( A ) * conjx( x )

  where A is an m x b_n matrix, x is a vector of length b_n, y is a vector
  of length m, and alpha is a scalar. The operation is performed as a series
  of fused axpyv operations, and therefore A should be column-stored.

  Parameters:

  - conja:  Compute with conjugated values of A?
  - conjx:  Compute with conjugated values of x?
  - m:      The number of rows in matrix A.
  - b_n:    The number of columns in matrix A. Must be equal to or less than
            the fusing factor.
  - alpha:  The address of a scalar.
  - a:      The address of matrix A.
  - inca:   The row stride of A. inca should be unit unless the
            implementation makes special accomodation for non-unit values.
  - lda:    The column stride of A.
  - x:      The address of vector x.
  - incx:   The vector increment of x.
  - y:      The address of vector y.
  - incy:   The vector increment of y. incy should be unit unless the
            implementation makes special accomodation for non-unit values.

  This template code calls the reference implementation if any of the
  following conditions are true:

  - Either of the strides inca or incy is non-unit.
  - The address of A, the second column of A, and y are unaligned with
    different offsets.

  If the first/second columns of A and address of y are aligned, or unaligned
  by the same offset, then optimized code can be used for the bulk of the
  computation. This template shows how the front-edge case can be handled so
  that the remaining computation is aligned. (This template guarantees
  alignment in the main loops to be BLIS_SIMD_ALIGN_SIZE.)

  Additional things to consider:

  - When optimizing, you should fully unroll the loops over b_n. This is the
    dimension across which we are fusing axpyv operations.
  - This template code chooses to call the reference implementation whenever
    b_n is less than the fusing factor, so as to avoid having to handle edge
    cases. One may choose to optimize this edge case, if desired.
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
	const siz_t type_size       = sizeof( *a );

	dcomplex*   ap[ bli_zaxpyf_fusefac ];
	dcomplex*   xp[ bli_zaxpyf_fusefac ];
	dcomplex*   yp;

	dcomplex    alpha_x[ bli_zaxpyf_fusefac ];

	bool        use_ref         = FALSE;

	dim_t       m_pre           = 0;
	dim_t       m_iter;
	dim_t       m_left;

	dim_t       off_a, off_a2, off_y;
	dim_t       i, j;


	// Return early if possible.
	if ( bli_zero_dim2( m, b_n ) ) return;

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( b_n < bli_zaxpyf_fusefac )
	{
		use_ref = TRUE;
	}
	else if ( bli_has_nonunit_inc3( inca, incx, incy ) )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( a,     BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( a+lda, BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( y,     BLIS_SIMD_ALIGN_SIZE ) )
	{
		use_ref = TRUE;

		// If a, the second column of a, and y are unaligned by the same
		// offset, then we can still use an implementation that depends on
		// alignment for most of the operation.
		off_a  = bli_offset_from_alignment( a,     BLIS_SIMD_ALIGN_SIZE );
		off_a2 = bli_offset_from_alignment( a+lda, BLIS_SIMD_ALIGN_SIZE );
		off_y  = bli_offset_from_alignment( y,     BLIS_SIMD_ALIGN_SIZE );

		if ( off_a == off_y && off_a == off_a2 )
		{
			use_ref = FALSE;
			m_pre   = off_a / type_size;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		zaxpyf_ft f = bli_zaxpyf_template_ref;

		f
		(
		  conja,
		  conjx,
		  m,
		  b_n,
		  alpha,
		  a, inca, lda,
		  x, incx,
		  y, incy,
		  cntx
		);
        return;
	}


	// Compute the number of unrolled and leftover (edge) iterations.
	m_iter = ( m - m_pre ) / n_elem_per_iter;
	m_left = ( m - m_pre ) % n_elem_per_iter;


	// Initialize pointers into the columns of A and elements of x.
	for ( j = 0; j < b_n; ++j )
	{
		ap[ j ] = a + (j  )*lda;
		xp[ j ] = x + (j  )*incx;
	}
	yp = y;


	// Load elements of x or conj(x) into alpha_x and scale by alpha.
	if ( bli_is_noconj( conjx ) )
	{
		for ( j = 0; j < b_n; ++j )
		{
			bli_tcopys( z,z, *xp[ j ], alpha_x[ j ] );
			bli_tscals( z,z,z, *alpha, alpha_x[ j ] );
		}
	}
	else // if ( bli_is_conj( conjx ) )
	{
		for ( j = 0; j < b_n; ++j )
		{
			bli_tcopyjs( z,z, *xp[ j ], alpha_x[ j ] );
			bli_tscals( z,z,z, *alpha, alpha_x[ j ] );
		}
	}

	// Iterate over rows of A and y to compute:
	//   y += conja( A )*conjx( x );
	if ( bli_is_noconj( conja ) )
	{
		// Compute front edge cases if a and y were unaligned.
		for ( i = 0; i < m_pre; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_taxpys( z,z,z,z, alpha_x[ j ], *ap[ j ], *yp );

				ap[ j ] += 1;
			}
			yp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of alpha_x should be loaded once prior to the m_iter
		// loop, and the b_n loop should be fully unrolled. The addresses in
		// ap[] and yp are guaranteed to be aligned to
		// BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < m_iter; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_taxpys( z,z,z,z, alpha_x[ j ], *ap[ j ], *yp );

				ap[ j ] += n_elem_per_iter;
			}
			yp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < m_left; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_taxpys( z,z,z,z, alpha_x[ j ], *ap[ j ], *yp );

				ap[ j ] += 1;
			}
			yp += 1;
		}
	}
	else // if ( bli_is_conj( conja ) )
	{
		// Compute front edge cases if a and y were unaligned.
		for ( i = 0; i < m_pre; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_taxpyjs( z,z,z,z, alpha_x[ j ], *ap[ j ], *yp );

				ap[ j ] += 1;
			}
			yp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of alpha_x should be loaded once prior to the m_iter
		// loop, and the b_n loop should be fully unrolled. The addresses in
		// ap[] and yp are guaranteed to be aligned to
		// BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < m_iter; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_taxpyjs( z,z,z,z, alpha_x[ j ], *ap[ j ], *yp );

				ap[ j ] += n_elem_per_iter;
			}
			yp += n_elem_per_iter;
		}

		// Compute tail edge cases.
		for ( i = 0; i < m_left; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_taxpyjs( z,z,z,z, alpha_x[ j ], *ap[ j ], *yp );

				ap[ j ] += 1;
			}
			yp += 1;
		}
	}

}

