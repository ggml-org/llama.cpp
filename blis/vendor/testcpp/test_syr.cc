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
#define ALPHA 1.0
#define N 6

/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T >
void ref_syr(int64_t n, 
    T * alpha,
    T *X,
    T *A )

{
   obj_t obj_a;
   obj_t obj_x;
   obj_t obj_alpha;
   num_t dt;
  
   if(is_same<T, float>::value)
       dt = BLIS_FLOAT;
   else if(is_same<T, double>::value)
    	dt = BLIS_DOUBLE;
   else if(is_same<T, complex<float>>::value)
    	dt = BLIS_SCOMPLEX;
   else if(is_same<T, complex<double>>::value)
    	dt = BLIS_DCOMPLEX;

   bli_obj_create_with_attached_buffer( dt, 1, 1, alpha, 1,1,&obj_alpha );
   bli_obj_create_with_attached_buffer( dt, n, n, A, 1, n, &obj_a );
   bli_obj_create_with_attached_buffer( dt, n, 1, X, 1, n,&obj_x );

   bli_obj_set_struc( BLIS_SYMMETRIC, &obj_a );
   bli_obj_set_uplo( BLIS_LOWER, &obj_a);
   bli_syr( &obj_alpha,
            &obj_x,
            &obj_a );
	
}
template< typename T >
void test_syr(  )
{
    T *A, *X, *A_ref;
    T alpha;
    int n;
    int lda, incx, lda_ref;

    alpha = ALPHA;
    n = N;

    lda = n;
    lda_ref = n;
    incx = 1;
     
    srand (time(NULL));
    allocate_init_buffer(A , n , n);
    allocate_init_buffer(X , n , 1);
    copy_buffer(A, A_ref , n ,n);

#ifdef PRINT
    printmatrix(A, lda ,n,n, (char *)"A");
    printvector(X, n,(char *) "X");
#endif    
	blis::syr(
	    CblasColMajor,
	    CblasLower,
            n,
	    alpha,
            X,
            incx,
            A,
            lda
            );

#ifdef PRINT
    printmatrix(A, lda , n , n,(char *) "A output");
#endif
    ref_syr(n, &alpha, X, A_ref); 
#ifdef PRINT
    printmatrix(A_ref, lda , n, n, (char *)"A ref output");
#endif
     if(computeErrorM(lda, lda_ref, n, n, A, A_ref )==1)
             printf("%s TEST FAIL\n" ,__PRETTY_FUNCTION__);
     else
             printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);


    delete[]( A     );
    delete[]( X     );
    delete[]( A_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_syr<double>( );
    test_syr<float>( );
    return 0;

}
