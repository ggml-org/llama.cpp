/*

   BLISPP
   C++ test driver for BLIS CPP symm routine and reference blis symm routine.

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
#define BETA 0.0
#define M 5
#define N 5
/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T >
void ref_symm(int64_t m, int64_t n,
 //   side_t side,
    T * alpha,
    T *A,
    T *B,
    T * beta,
    T *C )

{
   obj_t obj_a, obj_b, obj_c;
   obj_t obj_alpha, obj_beta;
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
   bli_obj_create_with_attached_buffer( dt, 1, 1, beta, 1,1,&obj_beta );
   bli_obj_create_with_attached_buffer( dt, m, m, A, 1,m,&obj_a );
   bli_obj_create_with_attached_buffer( dt, m, n, B, 1,n,&obj_b );
   bli_obj_create_with_attached_buffer( dt, m, n, C, 1,m,&obj_c );

   bli_obj_set_struc( BLIS_SYMMETRIC, &obj_a );
   bli_obj_set_uplo( BLIS_LOWER, &obj_a );
   bli_symm( BLIS_LEFT,
	     &obj_alpha,
             &obj_a,
             &obj_b,
             &obj_beta,
             &obj_c );
	
}
template< typename T >
void test_symm(  ) 
{
    T *A, *B, *C, *C_ref;
    T alpha, beta;
    int m,n;
    int    lda, ldb, ldc, ldc_ref;

    alpha = ALPHA;
    beta = BETA;
    m = M;
    n = N;

    lda = m;
    ldb = n;
    ldc	= m;
    ldc_ref = m;

    srand (time(NULL));
    allocate_init_buffer(A , m , m);
    allocate_init_buffer(B , m , n);
    allocate_init_buffer(C , m , n);
    copy_buffer(C, C_ref , m ,n);

#ifdef PRINT
    printmatrix(A, lda ,m,m, (char *)"A");
    printmatrix(B, ldb ,m,n, (char *)"B");
    printmatrix(C, ldc ,m,n, (char *)"C");
#endif
	blis::symm(
	    CblasColMajor,
	    CblasLeft,
	    CblasLower,
            m,
            n,
	    alpha,	
            A,
            lda,
            B,
            ldb,
	    beta,
            C,
            ldc
            );

#ifdef PRINT
    printmatrix(C, ldc ,m,n, (char *)"C output");
#endif
      // ref_symm(m, n, side, &alpha, A, B, &beta, C_ref);
       ref_symm(m, n,  &alpha, A, B, &beta, C_ref);

#ifdef PRINT
    printmatrix(C_ref, ldc_ref ,m,n, (char *)"C ref output");
#endif
     if(computeErrorM(ldc, ldc_ref, m, n, C, C_ref )==1)
         printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__ );
     else
         printf("%s TEST PASS\n" , __PRETTY_FUNCTION__ );


    delete[]( A     );
    delete[]( B     );
    delete[]( C     );
    delete[]( C_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_symm<double>( );
    test_symm<float>( );
    test_symm<complex<float>>( );
    test_symm<complex<double>>( );
    return 0;

}	
