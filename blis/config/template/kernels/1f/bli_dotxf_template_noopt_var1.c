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


void bli_zdotxf_template_noopt
     (
       conj_t             conjat,
       conj_t             conjx,
       dim_t              m,
       dim_t              b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict beta,
       dcomplex* restrict y, inc_t incy,
       cntx_t*            cntx
     )
{
/*
  Template dotxf kernel implementation

  This function contains a template implementation for a double-precision
  complex kernel, coded in C, which can serve as the starting point for one
  to write an optimized kernel on an arbitrary architecture. (We show a
  template implementation for only double-precision complex because the
  templates for the other three floating-point types would be similar, with
  the real instantiations being noticeably simpler due to the disappearance
  of conjugation in the real domain.)

  This kernel performs the following gemv-like operation:

    y := beta * y + alpha * conjat( A^T ) * conjx( x )

  where A is an m x b_n matrix, x is a vector of length m, y is a vector
  of length b_n, and alpha and beta are scalars. The operation is performed
  as a series of fused dotxv operations, and therefore A should be column-
  stored.

  Parameters:

  - conjat: Compute with conjugated values of A^T?
  - conjx:  Compute with conjugated values of x?
  - m:      The number of rows in matrix A.
  - b_n:    The number of columns in matrix A. Must be equal to or less than
            the fusing factor.
  - alpha:  The address of the scalar to be applied to A*x.
  - a:      The address of matrix A.
  - inca:   The row stride of A. inca should be unit unless the
            implementation makes special accomodation for non-unit values.
  - lda:    The column stride of A.
  - x:      The address of vector x.
  - incx:   The vector increment of x. incx should be unit unless the
            implementation makes special accomodation for non-unit values.
  - beta:   The address of the scalar to be applied to y.
  - y:      The address of vector y.
  - incy:   The vector increment of y.

  This template code calls the reference implementation if any of the
  following conditions are true:

  - Either of the strides inca or incx is non-unit.
  - The address of A, the second column of A, and x are unaligned with
    different offsets.

  If the first/second columns of A and address of x are aligned, or unaligned
  by the same offset, then optimized code can be used for the bulk of the
  computation. This template shows how the front-edge case can be handled so
  that the remaining computation is aligned. (This template guarantees
  alignment in the main loops to be BLIS_SIMD_ALIGN_SIZE.)

  Additional things to consider:

  - When optimizing, you should fully unroll the loops over b_n. This is the
    dimension across which we are fusing dotxv operations.
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
	const siz_t type_size       = sizeof( *x );

	dcomplex*   ap[ bli_zdotxf_fusefac ];
	dcomplex*   xp;
	dcomplex*   yp[ bli_zdotxf_fusefac ];

	dcomplex    Atx[ bli_zdotxf_fusefac ];

	bool        use_ref         = FALSE;

	dim_t       m_pre           = 0;
	dim_t       m_iter;
	dim_t       m_left;

	dim_t       off_a, off_a2, off_x;
	dim_t       i, j;

	conj_t      conjat_use;


	// Return early if possible.
	if ( bli_zero_dim1( b_n ) ) return;

	// If the vector lengths are zero, scale r by beta and return.
	if ( bli_zero_dim1( m ) )
	{
		bli_zscalv_ex
		(
		  BLIS_NO_CONJUGATE,
		  b_n,
		  beta,
		  y, incy,
		  cntx
		);
		return;
	}

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( b_n < bli_zdotxf_fusefac )
	{
		use_ref = TRUE;
	}
	else if ( bli_has_nonunit_inc2( inca, incx ) )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( a,     BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( a+lda, BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( x,     BLIS_SIMD_ALIGN_SIZE ) )
	{
		use_ref = TRUE;

		// If a, the second column of a, and x are unaligned by the same
		// offset, then we can still use an implementation that depends on
		// alignment for most of the operation.
		off_a  = bli_offset_from_alignment( a,     BLIS_SIMD_ALIGN_SIZE );
		off_a2 = bli_offset_from_alignment( a+lda, BLIS_SIMD_ALIGN_SIZE );
		off_x  = bli_offset_from_alignment( x,     BLIS_SIMD_ALIGN_SIZE );

		if ( off_a == off_a2 && off_a == off_x )
		{
			use_ref = FALSE;
			m_pre   = off_x / type_size;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		zdotxf_ft f = bli_zdotxf_template_ref;

		f
		(
		  conjat,
		  conjx,
		  m,
		  b_n,
		  alpha,
		  a, inca, lda,
		  x, incx,
		  beta,
		  y, incy,
		  cntx
		);
        return;
	}


	// Compute the number of unrolled and leftover (edge) iterations.
	m_iter = ( m - m_pre ) / n_elem_per_iter;
	m_left = ( m - m_pre ) % n_elem_per_iter;


	// Initialize pointers into the rows of A and elements of y.
	for ( i = 0; i < b_n; ++i )
	{
		ap[ i ] = a + (i  )*lda;
		yp[ i ] = y + (i  )*incy;
	}
	xp = x;


	// Initialize our accumulators to zero.
	for ( i = 0; i < b_n; ++i )
	{
		bli_tset0s( z, Atx[ i ] );
	}


	conjat_use = conjat;

	// If x must be conjugated, we compute the result indirectly by first
	// toggling the effective conjugation of A and then conjugating the
	// resulting product A^T*x.
	if ( bli_is_conj( conjx ) )
		bli_toggle_conj( &conjat_use );


	// Iterate over columns of A and rows of x to compute:
	//   Atx = conjat_use( A^T ) * x;
	if ( bli_is_noconj( conjat_use ) )
	{
		// Compute front edge cases if A and y were unaligned.
		for ( j = 0; j < m_pre; ++j )
		{
			for ( i = 0; i < b_n; ++i )
			{
				bli_tdots( z,z,z,z, *ap[ i ], *xp, Atx[ i ] );

				ap[ i ] += 1;
			}
			xp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of Atx should be kept in registers, and the b_n loop
		// should be fully unrolled. The addresses in ap[] and xp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( j = 0; j < m_iter; ++j )
		{
			for ( i = 0; i < b_n; ++i )
			{
				bli_tdots( z,z,z,z, *ap[ i ], *xp, Atx[ i ] );

				ap[ i ] += n_elem_per_iter;
			}
			xp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( j = 0; j < m_left; ++j )
		{
			for ( i = 0; i < b_n; ++i )
			{
				bli_tdots( z,z,z,z, *ap[ i ], *xp, Atx[ i ] );

				ap[ i ] += 1;
			}
			xp += 1;
		}
	}
	else // if ( bli_is_conj( conjat_use ) )
	{
		// Compute front edge cases if A and y were unaligned.
		for ( j = 0; j < m_pre; ++j )
		{
			for ( i = 0; i < b_n; ++i )
			{
				bli_tdotjs( z,z,z,z, *ap[ i ], *xp, Atx[ i ] );

				ap[ i ] += 1;
			}
			xp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of Atx should be kept in registers, and the b_n loop
		// should be fully unrolled. The addresses in ap[] and xp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( j = 0; j < m_iter; ++j )
		{
			for ( i = 0; i < b_n; ++i )
			{
				bli_tdotjs( z,z,z,z, *ap[ i ], *xp, Atx[ i ] );

				ap[ i ] += n_elem_per_iter;
			}
			xp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( j = 0; j < m_left; ++j )
		{
			for ( i = 0; i < b_n; ++i )
			{
				bli_tdotjs( z,z,z,z, *ap[ i ], *xp, Atx[ i ] );

				ap[ i ] += 1;
			}
			xp += 1;
		}
	}


	// If conjugation on y was requested, we induce it by conjugating
	// the contents of Atx.
	if ( bli_is_conj( conjx ) )
	{
		for ( i = 0; i < b_n; ++i )
		{
			bli_tconjs( z, Atx[ i ] );
		}
	}


	// Scale the Atx product by alpha and accumulate into y after
	// scaling by beta.
	for ( i = 0; i < b_n; ++i )
	{
		bli_tscals( z,z,z, *beta, *yp[ i ] );
		bli_taxpys( z,z,z,z, *alpha, Atx[ i ], *yp[ i ] );
	}
}

