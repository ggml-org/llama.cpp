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


void bli_zdotxaxpyf_template_noopt
     (
       conj_t             conjat,
       conj_t             conja,
       conj_t             conjw,
       conj_t             conjx,
       dim_t              m,
       dim_t              b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict w, inc_t incw,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict beta,
       dcomplex* restrict y, inc_t incy,
       dcomplex* restrict z, inc_t incz,
       cntx_t*            cntx
     )

{
/*
  Template dotxaxpyf kernel implementation

  This function contains a template implementation for a double-precision
  complex kernel, coded in C, which can serve as the starting point for one
  to write an optimized kernel on an arbitrary architecture. (We show a
  template implementation for only double-precision complex because the
  templates for the other three floating-point types would be similar, with
  the real instantiations being noticeably simpler due to the disappearance
  of conjugation in the real domain.)

  This kernel performs the following two gemv-like operations:

    y := beta * y + alpha * conjat( A^T ) * conjw( w )
    z :=        z + alpha * conja( A )    * conjx( x )

  where A is an m x b_n matrix, x and y are vector of length b_n, w and z
  are vectors of length m, and alpha and beta are scalars. The operation
  fuses a dotxf and an axpyf operation, and therefore A should be column-
  stored.

  Parameters:

  - conjat: Compute with conjugated values of A^T?
  - conja:  Compute with conjugated values of A?
  - conjw:  Compute with conjugated values of w?
  - conjx:  Compute with conjugated values of x?
  - m:      The number of rows in matrix A.
  - b_n:    The number of columns in matrix A. Must be equal to or less than
            the fusing factor.
  - alpha:  The address of the scalar to be applied to A^T*w and A*x.
  - a:      The address of matrix A.
  - inca:   The row stride of A. inca should be unit unless the
            implementation makes special accomodation for non-unit values.
  - lda:    The column stride of A.
  - w:      The address of vector w.
  - incw:   The vector increment of w. incw should be unit unless the
            implementation makes special accomodation for non-unit values.
  - x:      The address of vector x.
  - incx:   The vector increment of x.
  - beta:   The address of the scalar to be applied to y.
  - y:      The address of vector y.
  - incy:   The vector increment of y.
  - z:      The address of vector z.
  - incz:   The vector increment of z. incz should be unit unless the
            implementation makes special accomodation for non-unit values.

  This template code calls the reference implementation if any of the
  following conditions are true:

  - Any of the strides inca, incw, or incz is non-unit.
  - The address of A, the second column of A, w, and z are unaligned with
    different offsets.

  If the first/second rows of A and addresses of w and z are aligned, or
  unaligned by the same offset, then optimized code can be used for the bulk
  of the computation. This template shows how the front-edge case can be
  handled so that the remaining computation is aligned. (This template
  guarantees alignment in the main loops to be BLIS_SIMD_ALIGN_SIZE.)

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
	const siz_t type_size       = sizeof( *a );

	dcomplex*   ap[ bli_zdotxaxpyf_fusefac ];
	dcomplex*   xp[ bli_zdotxaxpyf_fusefac ];
	dcomplex*   yp[ bli_zdotxaxpyf_fusefac ];
	dcomplex*   wp;
	dcomplex*   zp;

	dcomplex    At_w[ bli_zdotxaxpyf_fusefac ];
	dcomplex    alpha_x[ bli_zdotxaxpyf_fusefac ];

	bool        use_ref         = FALSE;

	dim_t       m_pre           = 0;
	dim_t       m_iter;
	dim_t       m_left;

	dim_t       off_a, off_a2, off_w, off_z;
	dim_t       i, j;

	conj_t      conjat_use;


	// Return early if possible.
	if ( bli_zero_dim2( m, b_n ) ) return;

	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( b_n < bli_zdotxaxpyf_fusefac )
	{
		use_ref = TRUE;
	}
	else if ( bli_has_nonunit_inc3( inca, incw, incz ) )
	{
		use_ref = TRUE;
	}
	else if ( bli_is_unaligned_to( a,     BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( a+lda, BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( w,     BLIS_SIMD_ALIGN_SIZE ) ||
	          bli_is_unaligned_to( z,     BLIS_SIMD_ALIGN_SIZE ) )
	{
		use_ref = TRUE;

		// If a, the second column of a, w, and z are unaligned by the same
		// offset, then we can still use an implementation that depends on
		// alignment for most of the operation.
		off_a  = bli_offset_from_alignment( a,     BLIS_SIMD_ALIGN_SIZE );
		off_a2 = bli_offset_from_alignment( a+lda, BLIS_SIMD_ALIGN_SIZE );
		off_w  = bli_offset_from_alignment( w,     BLIS_SIMD_ALIGN_SIZE );
		off_z  = bli_offset_from_alignment( z,     BLIS_SIMD_ALIGN_SIZE );

		if ( off_a == off_a2 && off_a == off_w && off_a == off_z )
		{
			use_ref = FALSE;
			m_pre   = off_a / type_size;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		zdotxaxpyf_ft f = bli_zdotxaxpyf_template_ref;

		f
		(
		  conjat,
		  conja,
		  conjw,
		  conjx,
		  m,
		  b_n,
		  alpha,
		  a, inca, lda,
		  w, incw,
		  x, incx,
		  beta,
		  y, incy,
		  z, incz,
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
		yp[ j ] = y + (j  )*incy;
	}
	wp = w;
	zp = z;

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

	// Initialize our accumulators to zero.
	for ( j = 0; j < b_n; ++j )
	{
		bli_tset0s( z, At_w[ j ] );
	}


	conjat_use = conjat;

	// If w must be conjugated, we compute the result indirectly by first
	// toggling the effective conjugation of At and then conjugating the
	// resulting dot products.
	if ( bli_is_conj( conjw ) )
		bli_toggle_conj( &conjat_use );


	// Iterate over the columns of A and elements of w and z to compute:
	//   y = beta * y + alpha * conjat( A^T ) * conjw( w );
    //   z =        z + alpha * conja( A )    * conjx( x );
	// where A is m x b_n.
	if ( bli_is_noconj( conja ) && bli_is_noconj( conjat_use ) )
	{
		// Compute front edge cases if A, w, and z were unaligned.
		for ( i = 0; i < m_pre; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdots( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdots( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of alpha_x should be loaded once prior to the m_iter
		// loop, At_w should be kept in registers, and the b_n loop should
		// be fully unrolled. The addresses in ap[], wp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < m_iter; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdots( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdots( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += n_elem_per_iter;
			}
			wp += n_elem_per_iter; zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < m_left; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdots( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdots( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}
	}
	else if ( bli_is_noconj( conja ) && bli_is_conj( conjat_use ) )
	{
		// Compute front edge cases if A, w, and z were unaligned.
		for ( i = 0; i < m_pre; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdotjs( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdots( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of alpha_x should be loaded once prior to the m_iter
		// loop, At_w should be kept in registers, and the b_n loop should
		// be fully unrolled. The addresses in ap[], wp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < m_iter; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdotjs( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdots( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += n_elem_per_iter;
			}
			wp += n_elem_per_iter; zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < m_left; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdotjs( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdots( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}
	}
	else if ( bli_is_conj( conja ) && bli_is_noconj( conjat_use ) )
	{
		// Compute front edge cases if A, w, and z were unaligned.
		for ( i = 0; i < m_pre; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdots( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdotjs( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of alpha_x should be loaded once prior to the m_iter
		// loop, At_w should be kept in registers, and the b_n loop should
		// be fully unrolled. The addresses in ap[], wp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < m_iter; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdots( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdotjs( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += n_elem_per_iter;
			}
			wp += n_elem_per_iter; zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < m_left; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdots( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdotjs( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}
	}
	else if ( bli_is_conj( conja ) && bli_is_conj( conjat_use ) )
	{
		// Compute front edge cases if A, w, and z were unaligned.
		for ( i = 0; i < m_pre; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdotjs( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdotjs( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}

		// The bulk of the operation is executed here. For best performance,
		// the elements of alpha_x should be loaded once prior to the m_iter
		// loop, At_w should be kept in registers, and the b_n loop should
		// be fully unrolled. The addresses in ap[], wp, and zp are
		// guaranteed to be aligned to BLIS_SIMD_ALIGN_SIZE.
		for ( i = 0; i < m_iter; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdotjs( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdotjs( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += n_elem_per_iter;
			}
			wp += n_elem_per_iter; zp += n_elem_per_iter;
		}

		// Compute tail edge cases, if applicable.
		for ( i = 0; i < m_left; ++i )
		{
			for ( j = 0; j < b_n; ++j )
			{
				bli_tdotjs( z,z,z,z, *ap[ j ], *wp, At_w[ j ] );
				bli_tdotjs( z,z,z,z, *ap[ j ], alpha_x[ j ], *zp );

				ap[ j ] += 1;
			}
			wp += 1; zp += 1;
		}
	}


	// If conjugation on w was requested, we induce it by conjugating
	// the contents of At_w.
	if ( bli_is_conj( conjw ) )
	{
		for ( j = 0; j < b_n; ++j )
		{
			bli_tconjs( z, At_w[ j ] );
		}
	}

	// Scale the At_w product by alpha and accumulate into y after
	// scaling by beta.
	for ( j = 0; j < b_n; ++j )
	{
		bli_tscals( z,z,z, *beta, *yp[ j ] );
		bli_taxpys( z,z,z,z, *alpha, At_w[ j ], *yp[ j ] );
	}
}

