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
#define ALPHA 0.5

/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename TA,typename TB>
void ref_scal(int64_t n,
    TA * alpha,
    TB *X
     )

{
   obj_t obj_x;
   obj_t obj_alpha;
   num_t dt_x , dt_alpha;
   if(is_same<TB , float>::value)
     dt_x = BLIS_FLOAT;
   else if(is_same<TB , double>::value)
     dt_x = BLIS_DOUBLE;
   else if(is_same<TB , complex<float>>::value)
     dt_x = BLIS_SCOMPLEX;
   else if(is_same<TB , complex<double>>::value)
     dt_x = BLIS_DCOMPLEX;
  
   if(is_same<TA , float>::value)
     dt_alpha = BLIS_FLOAT;
   else if(is_same<TA , double>::value)
     dt_alpha = BLIS_DOUBLE;
   else if(is_same<TA , complex<float>>::value)
     dt_alpha = BLIS_SCOMPLEX;
   else if(is_same<TA , complex<double>>::value)
     dt_alpha = BLIS_DCOMPLEX;

   bli_obj_create_with_attached_buffer( dt_alpha, 1, 1, alpha, 1,1,&obj_alpha );
   bli_obj_create_with_attached_buffer( dt_x, n, 1, X, 1, n,&obj_x );

   bli_scalv(&obj_alpha,
            &obj_x
	     );
	
}
template< typename TA, typename TB>
void test_scal()
{
    TB *X, *X_ref;
    TA alpha = ALPHA;
    int n;
    int incx;

    n = N;

    incx = 1;
    srand (time(NULL));
    allocate_init_buffer(X , n , 1);
    copy_buffer(X, X_ref , n ,1);

#ifdef PRINT
    printvector(X, n, (char *)"X");
#endif     
	blis::scal<TA, TB>(
            n,
            alpha,
            X,
            incx
	    );

#ifdef PRINT
    printvector(X, n, (char *)"X output");
#endif
       ref_scal(n ,  &alpha , X_ref ); 

#ifdef PRINT
    printvector(X_ref, n, (char *)"X ref output");
#endif
     if(computeErrorV(incx, incx , n, X, X_ref )==1)
        printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__);
     else
        printf("%s TEST PASS\n" , __PRETTY_FUNCTION__);


    delete[]( X     );
    delete[]( X_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_scal<float , float>( );
    test_scal<double , double>( );
    test_scal<std::complex<float> , std::complex<float>>( );
    test_scal<std::complex<double> , std::complex<double>>( );
    test_scal<float , std::complex<float>>( );
    test_scal<double , std::complex<double>>( );
    return 0;

}
