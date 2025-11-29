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
    - Neither the name of The University of Texas nor the names of its
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

#include <stdio.h>
#include "blis.h"

int main( int argc, char** argv )
{
	obj_t a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;
	obj_t v1, v2;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;


	//
	// This file demonstrates the basics of creating objects in BLIS,
	// inspecting their basic properties, and printing matrix objects.
	//


	//
	// Example 1: Create an object containing a 4x3 matrix of double-
	//            precision real elements stored in column-major order.
	//

	// The matrix dimensions are m = 4 and n = 3. We choose to use column
	// storage (often called column-major storage) and thus we specify
	// that the row stride ("rs" for short) argument is 1 and the column
	// stride ("cs" for short) argument is equal to m = 4. In column
	// storage, cs is known as the leading dimension.
	dt = BLIS_DOUBLE; m  = 4; n  = 3;
	                  rs = 1; cs = 4; 
	bli_obj_create( dt, m, n, rs, cs, &a1 );

	// If cs is greater than m, then extra rows (in this case, two) will
	// be allocated beyond the lower edge of the matrix. Sometimes this
	// is desireable for alignment purposes.
	dt = BLIS_DOUBLE; m  = 4; n  = 3;
	                  rs = 1; cs = 6; 
	bli_obj_create( dt, m, n, rs, cs, &a2 );


	//
	// Example 2: Create an object containing a 4x3 matrix of double-
	//            precision real elements stored in row-major order.
	//

	// Here, we choose to use row storage (often called row-major storage)
	// and thus we specify that the cs is 1 and rs is equal to n = 3. In
	// row storage, the leading dimension corresponds to rs.
	dt = BLIS_DOUBLE; m  = 4; n  = 3;
	                  rs = 3; cs = 1; 
	bli_obj_create( dt, m, n, rs, cs, &a3 );

	// As with the second example, we can cause extra columns (in this
	// case, five) to be allocated beyond the right edge of the matrix.
	dt = BLIS_DOUBLE; m  = 4; n  = 3;
	                  rs = 8; cs = 1; 
	bli_obj_create( dt, m, n, rs, cs, &a4 );


	//
	// Example 3: Create objects using other floating-point datatypes.
	//

	// Examples of using the other floating-point datatypes.
	                  m  = 4; n  = 3;
	                  rs = 1; cs = 4; 
	bli_obj_create( BLIS_FLOAT,    m, n, rs, cs, &a5 );
	bli_obj_create( BLIS_SCOMPLEX, m, n, rs, cs, &a6 );
	bli_obj_create( BLIS_DCOMPLEX, m, n, rs, cs, &a7 );


	//
	// Example 4: Create objects using default (column) storage so that
	//            we avoid having to specify rs and cs manually.
	//

	// Specifying the row and column strides as zero, as is done here, is
	// a shorthand request for the default storage scheme, which is
	// currently (and always has been) column storage. When requesting the
	// default storage scheme with rs = cs = 0, BLIS may insert additional
	// padding for alignment purposes. So, the 3x8 matrix object created
	// below may end up having a row stride that is greater than 3. When
	// in doubt, query the value! 
	bli_obj_create( BLIS_FLOAT, 3, 5, 0, 0, &a8 );


	//
	// Example 5: Inspect object fields after creation to expose
	//            possible alignment/padding.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Let's inspect the amount of padding inserted for alignment. Note
	// the difference between the m dimension and the column stride.
	printf( "datatype            %s\n", bli_dt_string( bli_obj_dt( &a8 ) ) );
	printf( "datatype size       %d bytes\n", ( int )bli_dt_size( bli_obj_dt( &a8 ) ) );
	printf( "m dim (# of rows):  %d\n", ( int )bli_obj_length( &a8 ) );
	printf( "n dim (# of cols):  %d\n", ( int )bli_obj_width( &a8 ) );
	printf( "row stride:         %d\n", ( int )bli_obj_row_stride( &a8 ) );
	printf( "col stride:         %d\n", ( int )bli_obj_col_stride( &a8 ) );


	//
	// Example 6: Inspect object fields after creation of other floating-
	//            point datatypes.
	//

	printf( "\n#\n#  -- Example 6 --\n#\n\n" );

	bli_obj_create( BLIS_DOUBLE,   3, 5, 0, 0, &a9 );
	bli_obj_create( BLIS_SCOMPLEX, 3, 5, 0, 0, &a10);
	bli_obj_create( BLIS_DCOMPLEX, 3, 5, 0, 0, &a11 );

	printf( "datatype            %s\n", bli_dt_string( bli_obj_dt( &a9 ) ) );
	printf( "datatype size       %d bytes\n", ( int )bli_dt_size( bli_obj_dt( &a9 ) ) );
	printf( "m dim (# of rows):  %d\n", ( int )bli_obj_length( &a9 ) );
	printf( "n dim (# of cols):  %d\n", ( int )bli_obj_width( &a9 ) );
	printf( "row stride:         %d\n", ( int )bli_obj_row_stride( &a9 ) );
	printf( "col stride:         %d\n", ( int )bli_obj_col_stride( &a9 ) );

	printf( "\n" );
	printf( "datatype            %s\n", bli_dt_string( bli_obj_dt( &a10 ) ) );
	printf( "datatype size       %d bytes\n", ( int )bli_dt_size( bli_obj_dt( &a10 ) ) );
	printf( "m dim (# of rows):  %d\n", ( int )bli_obj_length( &a10 ) );
	printf( "n dim (# of cols):  %d\n", ( int )bli_obj_width( &a10 ) );
	printf( "row stride:         %d\n", ( int )bli_obj_row_stride( &a10 ) );
	printf( "col stride:         %d\n", ( int )bli_obj_col_stride( &a10 ) );

	printf( "\n" );
	printf( "datatype            %s\n", bli_dt_string( bli_obj_dt( &a11 ) ) );
	printf( "datatype size       %d bytes\n", ( int )bli_dt_size( bli_obj_dt( &a11 ) ) );
	printf( "m dim (# of rows):  %d\n", ( int )bli_obj_length( &a11 ) );
	printf( "n dim (# of cols):  %d\n", ( int )bli_obj_width( &a11 ) );
	printf( "row stride:         %d\n", ( int )bli_obj_row_stride( &a11 ) );
	printf( "col stride:         %d\n", ( int )bli_obj_col_stride( &a11 ) );


	//
	// Example 7: Initialize an object's elements to random values and then
	//            print the matrix.
	//

	printf( "\n#\n#  -- Example 7 --\n#\n\n" );

	// We can set matrices to random values. The default behavior of
	// bli_randm() is to use random values on the internval [-1,1].
	bli_randm( &a9 );

	// And we can also print the matrices associated with matrix objects.
	// Notice that the third argument is a printf()-style format specifier.
	// Any valid printf() format specifier can be passed in here, but you
	// still need to make sure that the specifier makes sense for the data
	// being printed. For example, you shouldn't use "%d" when printing
	// elements of type 'float'.
	bli_printm( "matrix 'a9' contents:", &a9, "% 4.3f", "" );


	//
	// Example 8: Randomize and then print from an object containing a complex
	//            matrix.
	//

	printf( "\n#\n#  -- Example 8 --\n#\n\n" );

	// When printing complex matrices, the same format specifier gets used
	// for both the real and imaginary parts.
	bli_randm( &a11 );
	bli_printm( "matrix 'a11' contents (complex):", &a11, "% 4.3f", "" );


	//
	// Example 9: Create, randomize, and print vector objects.
	//

	printf( "\n#\n#  -- Example 9 --\n#\n\n" );

	// Now let's create two vector objects--a row vector and a column vector.
	// (A vector object is like a matrix object, except that it has at least
	// one unit dimension (equal to one).
	bli_obj_create( BLIS_DOUBLE, 4, 1, 0, 0, &v1 );
	bli_obj_create( BLIS_DOUBLE, 1, 6, 0, 0, &v2 );

	// If we know the object is a vector, we can use bli_randv(), though
	// bli_randm() would work just as well, since any vector is also a matrix.
	bli_randv( &v1 );
	bli_randv( &v2 );

	// We can print vectors, too.
	bli_printm( "vector 'v1' contents:", &v1, "%5.1f", "" );
	bli_printm( "vector 'v2' contents:", &v2, "%5.1f", "" );


	// Free all of the objects we created.
	bli_obj_free( &a1 );
	bli_obj_free( &a2 );
	bli_obj_free( &a3 );
	bli_obj_free( &a4 );
	bli_obj_free( &a5 );
	bli_obj_free( &a6 );
	bli_obj_free( &a7 );
	bli_obj_free( &a8 );
	bli_obj_free( &a9 );
	bli_obj_free( &a10 );
	bli_obj_free( &a11 );
	bli_obj_free( &v1 );
	bli_obj_free( &v2 );

	return 0;
}

