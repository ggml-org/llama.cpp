/*

   BLISPP
   C++ test driver for BLIS CPP gemm routine and reference blis gemm routine.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

#include <complex>
#include <iostream>
#include "blis.hh"
#include "test.hh"

using namespace blis;
using namespace std;
//#define PRINT
#define N 6

/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T>
void ref_swap(int64_t n,
    T *X,
    T *Y
     )

{
   obj_t obj_x, obj_y;
   num_t dt;
  
   if(is_same<T, float>::value)
     dt = BLIS_FLOAT;
   else if(is_same<T, double>::value)
     dt = BLIS_DOUBLE;
   else if(is_same<T, complex<float>>::value)
     dt = BLIS_SCOMPLEX;
   else if(is_same<T, complex<double>>::value)
     dt = BLIS_DCOMPLEX;


   bli_obj_create_with_attached_buffer( dt, n, 1, X, 1, n,&obj_x );
   bli_obj_create_with_attached_buffer( dt, n, 1, Y, 1, n,&obj_y );

   bli_swapv( &obj_x,
	     &obj_y
	     );
	
}
template< typename T >
void test_swap( )
{
    T *X, *X_ref, *Y,*Y_ref;
    int n;
    int incx, incy;

    n = N;

    incx = 1;
    incy = 1;
     
    srand (time(NULL));
    allocate_init_buffer(X , n , 1);
    allocate_init_buffer(Y , n , 1);
    copy_buffer(X, X_ref , n ,1);
    copy_buffer(Y, Y_ref , n ,1);

#ifdef PRINT
    printvector(X, n, (char *)"X");
    printvector(Y, n, (char *)"Y");
#endif
	blis::swap(
            n,
            X,
            incx,
	    Y,
	    incy
	    );

#ifdef PRINT
    printvector(X, n, (char *)"X output");
    printvector(Y, n, (char *)"Y output"); 
#endif
    ref_swap(n , X_ref, Y_ref ); 

#ifdef PRINT
    printvector(X_ref, n, (char *)"X ref output");
    printvector(Y_ref, n, (char *)"Y ref output");
#endif
     if((computeErrorV(incy, incy,n, Y, Y_ref )==1)||(computeErrorV(incx, incx, n, X, X_ref )==1))
        printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
     else
        printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);


    delete[]( X     );
    delete[]( Y     );
    delete[]( Y_ref );
    delete[]( X_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_swap<float>( );
    test_swap<double>( );
    test_swap<complex<float>>( );
    test_swap<complex<double>>( );
    return 0;

}
